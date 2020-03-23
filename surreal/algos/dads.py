# Copyright 2019 ducandu GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from surreal.algos.rl_algo import RLAlgo
from surreal.algos.sac import SAC, SACConfig
from surreal.components import FIFOBuffer, NegLogLikelihoodLoss, Network, MixtureDistribution, Optimizer, Preprocessor
from surreal.config import AlgoConfig
from surreal.spaces import Dict, Float, Int, Space
from surreal.components.helper_functions import iterate_minibatches

class DADS(RLAlgo):
    """
    The DADS algorithm.
    [1] Dynamics-Aware Unsupervised Discovery of Skills - A. Sharma∗, S. Gu, S. Levine, V. Kumar, K. Hausman - Google Brain 2019
        Compare to "Algorithm 1" and "Algorithm 2" pseudocodes in paper.
    """
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.inference = False  # True=planning mode. False="supervised+intrinsic-reward+model-learning" mode. #TODO
        self.he = None  # Current step within He (total episode horizon).
        self.hz = None  # Current step within Hz (repeat horizon for one selected skill)

        self.preprocessor = Preprocessor.make(config.preprocessor)
        self.s = self.preprocessor(config.state_space.with_batch())  # preprocessed states
        self.a = config.action_space.with_batch()  # actions (a)
        self.ri = Float(main_axes=[("Episode Horizon", config.episode_horizon)])  # intrinsic rewards in He
        self.z = config.skill_space.with_batch()
        self.s_and_z = Dict(dict(s=self.s, z=self.z), main_axes="B")
        self.pi = Network.make(input_space=self.s_and_z, output_space=self.a, distributions=True,
                               **config.policy_network)
        self.q = Network.make(input_space=self.s_and_z, output_space=self.s,
                              distributions=dict(type="mixture", num_experts=config.num_q_experts), **config.q_network)
        self.B = FIFOBuffer(Dict(dict(s=self.s, z=self.z, a=self.a, t=bool)), config.episode_buffer_capacity,
                            next_record_setup=dict(s="s_"))
        self.SAC = SAC(config=config.sac_config, name="SAC-level0", existing_pi=self.pi)  # Low-level SAC.
        self.q_optimizer = Optimizer.make(config.supervised_optimizer)  # supervised model optimizer
        self.Lsup = NegLogLikelihoodLoss(distribution=MixtureDistribution(num_experts=config.num_q_experts))
        self.preprocessor.reset()
        self.q_loss = 0
        self.skill_divergence = 0
        self.skill_uniqueness = 0
        self.avg_ri = 0
        self.z_dists = []
        #TODO
        self.end_state_by_skill = {
            k: []
            for k in range(self.config.dim_skill_vectors)
        }

    def update(self, samples, time_percentage):
        # Update for K1 (num_steps_per_update_q) iterations on same batch.
        weights = self.q.get_weights(as_ref=True)
        s_ = samples["s_"] if self.config.q_predicts_states_diff is False else \
            tf.nest.map_structure(lambda s, s_: s_ - s, samples["s"], samples["s_"])
        for _ in range(self.config.num_steps_per_update_pi):
            for ms, mz, ms_ in iterate_minibatches(
                [samples["s"], samples["z"], s_],
                    self.config.minibatch_size_q):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(weights)
                    parameters = self.q(dict(s=ms, z=mz), parameters_only=True)
                    loss = self.Lsup(parameters, ms_)
                    self.q_loss = tf.reduce_mean(loss)
                    grads_and_weights = list(
                        zip(tape.gradient(self.q_loss, weights), weights))
                    self.q_optimizer.apply_gradients(
                        grads_and_weights, time_percentage=time_percentage)

        # Calculate intrinsic rewards.
        # Pull a batch of zs of size batch * (L - 1) (b/c 1 batch is the `z` of the sample (numerator's z)).
        batch_size = len(samples["s"])
        zs = tf.concat([samples["z"], self.z.sample(batch_size * (self.config.num_denominator_samples_for_ri - 1))], axis=0)
        s = tf.nest.map_structure(lambda s: tf.tile(s, [self.config.num_denominator_samples_for_ri] + ([1] * (len(s.shape) - 1))), samples["s"])
        s_ = tf.nest.map_structure(lambda s: tf.tile(s, [self.config.num_denominator_samples_for_ri] + ([1] * (len(s.shape) - 1))), samples["s_"])
        # Single (efficient) forward pass yielding s' likelihoods.
        all_s__llhs = tf.stack(tf.split(self.q(dict(s=s, z=zs), s_, log_likelihood=True), self.config.num_denominator_samples_for_ri))
        r = all_s__llhs[0] - tf.math.reduce_logsumexp(all_s__llhs, axis=0) + \
            tf.math.log(tf.cast(self.config.num_denominator_samples_for_ri, tf.float32))
        self.avg_ri = tf.reduce_mean(r)
        # Update RL-algo's policy (same as π) from our batch (using intrinsic rewards).
        z_exp = np.expand_dims(samples["z"], axis=-1)

        for _ in range(self.config.num_steps_per_update_pi):
            for ms, mz, ma, mr, ms_, mt in iterate_minibatches([
                    samples["s"], z_exp, samples["a"], r, samples["s_"],
                    samples["t"]
            ], self.config.minibatch_size_pi):
                self.SAC.update(  # SAC expects a simple pi(a|s) construction, hence the nested dictionaries
                    dict(s=dict(s=ms, z=mz),
                         a=ma,
                         r=mr,
                         s_=dict(s=ms_, z=mz),
                         t=mt), time_percentage)

        # a measure of how separated the action dists are, depending on condition z.
        if self.config.summaries is not None and 'skill_divergence' in self.config.summaries:
            action_dist = self.pi(dict(s=s, z=zs), distributions_only=True)
            skill_divergences = []
            for _s in range(batch_size):
                # action_dist_sel = action_dist[_s::batch_size]

                # a rough implementation of average divergence - see Andrea Sgarro, Informational divergence and the dissimilarity of probability distributions. Calcolo, 18, 293–302 (1981)
                # averaging over all z' samples: D_KL[π(a|s,z) || π(a|s,z')]
                skill_divergences.append(tf.reduce_mean(tfp.distributions.kl_divergence(action_dist[_s], action_dist[_s+batch_size::batch_size])))
            self.skill_divergence = tf.reduce_mean(skill_divergences)

        # count number of unique actions found per state (only makes sense in discrete setting)
        if self.config.summaries is not None and 'skill_uniqueness' in self.config.summaries:
            action_dets = self.pi(dict(s=s, z=zs), deterministic=True)
            skill_uniquenesses = []
            for _s in range(batch_size):
                skill_uniquenesses.append(tf.size(tf.unique(action_dets[_s::batch_size])[0]))
            self.skill_uniqueness = tf.reduce_mean(tf.cast(skill_uniquenesses, tf.float32))

    def event_episode_starts(self, event):
        # Initialize z, hz, and he if this hasn't happened yet.
        if self.z.value is None:
            self.z.assign(self.z.zeros(len(event.actor_slots)))
            self.hz = np.zeros(len(event.actor_slots), dtype=np.int32)
            self.he = np.zeros(len(event.actor_slots), dtype=np.int32)
        else: #TODO this is a hack.
            self.end_state_by_skill[self.z.value[0]].append(
                tuple([int(i) for i in event.env.processes[0][0]._get_x_y(np.argmax(self.s.value))]))
        # Sample new z at the trajectory's batch position.
        if self.inference is False:
            self.z.value[event.current_actor_slot] = self.z.sample()  # Sample a new skill from Space z and store it in z (assume uniform).
        # Reset preprocessor at actor's batch position.
        self.preprocessor.reset(batch_position=event.current_actor_slot)
        self.he.fill(0)
        self.hz.fill(0)
        event.env.reset_all()

    # Fill the buffer with M samples.
    def event_tick(self, event):
        # Preprocess state.
        s_ = self.preprocessor(event.s_)

        # if self.config.episode_horizon is not None and (self.he >= self.config.episode_horizon).any():
        #     # We have reached the end of the agent-enforced episode horizon -> reset.
        #     event.env.reset_all()  # Send reset request to env.
        #     return

        # if self.config.max_time_steps is not None and (self.he >= self.config.max_time_steps).any():
        #     # We have reached the end of the agent-enforced episode horizon -> reset.
        #     event.env.terminate()  # Send terminate request to env.
        #     return

        self.he += 1 # increment all actors

        # if not planning, we keep z fixed for a whole episode. Otherwise...
        if self.inference:
            # TODO currently there's no distinction between skill_horizon for different actors - so this is either all or nothing
            for i in event.actor_slots:
                # check whether it's time for a new skill
                if self.hz[i] > self.config.skill_horizon:
                    self.hz[i] = 0
                if self.hz[i] == 0:
                    self.z.value[i] = self.plan(s_)
            self.hz += 1

        # Add single(!) szas't-tuple to buffer.
        if event.actor_time_steps > 0:
            self.B.add_records(dict(s=self.s.value, z=self.z.value, a=self.a.value, t=event.t, s_=s_))
            if self.B.size == self.B.capacity:
                self.update(self.B.flush(), time_percentage=event.actor_time_steps /
                                                            (self.config.max_time_steps or event.env.max_time_steps))

        # Query policy for an action.
        a_ = self.pi(dict(s=s_, z=self.z.value)).numpy()

        # Send the new action back to the env.
        event.env.act(a_)

        # Store action and state for next tick.
        # NOTE: env.state updates on env.act, so s_, a_ is now the LAST (s,a) pair. Next tick, they will be added to self.B
        self.s.assign(s_)
        self.a.assign(a_)


    class Z_Dist:
        def __init__(self, skill_space):
            if isinstance(skill_space, Int):
                self.discrete = True
                self.n = skill_space.flat_dim_with_categories
                self.mu = np.ones(self.n) / self.n
                # self.cov = np.eye(n_a) # TODO? see MPPI
            else:
                raise "continuous Z not done yet"

        def sample(self, size=None):
            if self.discrete:
                #TODO - categorical?
                # return np.random.multivariate_normal(self.mu, self.cov, size),
                return np.random.choice(self.n, size=size, p=self.mu)

        def update(self, new_mu):
            self.mu = new_mu


    def plan(self, s_0):
        """
        MPPI (not finished yet)
        """

        if self.config.reward_function is None:
            # planning isn't going to help us
            return self.z.sample()

        remaining_horizon = self.config.plan_horizon
        if self.config.max_time_steps:
            remaining_horizon = min(remaining_horizon, self.config.max_time_steps - self.he)

        # fill out missing distributions for the path ahead of us
        while len(self.z_dists) < remaining_horizon:
            self.z_dists.append(self.Z_Dist(self.config.skill_space))

        reward = np.empty(self.config.mppi_num_samples)  # i.e. total return
        terminal = np.empty(self.config.mppi_num_samples, dtype=bool)

        # refine over R (num_iters) iterations
        for j in range(self.config.mppi_num_iters):
            reward.fill(0)
            terminal.fill(False)
            # Sample K (num_samples) plans of length Hp (len(p_d))
            z_seq = [
                zd.sample(self.config.mppi_num_samples)
                for zd in self.z_dists
            ]

            s_i = np.broadcast_to(s_0, [self.config.mppi_num_samples] + s_0.shape[1:])  # (K, state_dim)

            for z in z_seq:
                # evolve states along each of the K trajectories
                s_i = self.q(dict(s=s_i, z=z))  # (K, state_dim)
                r, t = self.config.reward_function(s_i) # (K,)
                reward += np.where(terminal, 0, r)
                terminal = np.logical_or(terminal, t)

            # update parameters
            total_sum = np.sum(np.exp(self.config.mppi_gamma * reward))
            for z, zd in zip(z_seq, self.z_dists):
                z_ = np.eye(self.config.skill_space.flat_dim_with_categories)[z]
                new_mu = np.sum(
                    np.exp(self.config.mppi_gamma * reward)[:, None] *
                    z_ / total_sum, axis=0)
                zd.update(new_mu) #TODO smoothing

        # finally, sample a skill for the first planning step
        return self.z_dists[0].sample()


class DADSConfig(AlgoConfig):
    """
    Config object for a DADS Algo.
    """
    def __init__(
            self, *,
            policy_network, q_network,
            state_space, action_space,
            sac_config,
            num_q_experts=4,  # 4 used in paper.
            q_predicts_states_diff=False,
            num_denominator_samples_for_ri=250,  # 50-500 used in paper
            dim_skill_vectors=10, discrete_skills=False, episode_horizon=200,
            skill_horizon=10,  # A.5 in paper
            plan_horizon=1,  # A.5 in paper
            preprocessor=None,
            supervised_optimizer=None,
            num_steps_per_update_q=1,
            num_steps_per_update_pi=1,
            minibatch_size_q=128,
            minibatch_size_pi=128,
            episode_buffer_capacity=200,
            max_time_steps=None,
            mppi_num_iters=10,  # from appendix A.5 in paper
            mppi_num_samples=100,  # "between 1 and 200"
            mppi_gamma=10,  # from appendix A.5 in paper
            reward_function=None,
            summaries=None
    ):
        """
        Args:
            policy_network (Network): The policy-network (pi) to use as a function approximator for the learnt policy.

            q_network (Network): The dynamics-network (q) to use as a function approximator for the learnt env
                dynamics. NOTE: Not to be confused with a Q-learning Q-net! In the paper, the dynamics function is
                called `q`, hence the same nomenclature here.

            state_space (Space): The state/observation Space.
            action_space (Space): The action Space.
            sac_config (SACConfig): The config for the internal SAC-Algo used to learn the skills using intrinsic rewards.

            num_q_experts (int): The number of experts used in the Mixture distribution output bz the q-network to
                predict the next state (s') given s (state) and z (skill vector).

            q_predicts_states_diff (bool): Whether the q-network predicts the different between s and s' rather than
                s' directly. Default: False.

            num_denominator_samples_for_ri (int): The number of samples to calculate for the denominator of the
                intrinsic reward function (`L` in the paper).

            dim_skill_vectors (int): The number of dimensions of the learnt skill vectors.
            discrete_skills (bool): Whether skill vectors are discrete (one-hot).
            episode_horizon (int): The episode horizon (He) to move within, when gathering episode samples.

            skill_horizon (Optional[int]): The horizon for which to use one skill vector (before sampling a new one).
                Default: Use value of `episode_horizon`.

            preprocessor (Preprocessor): The preprocessor (if any) to use.
            supervised_optimizer (Optimizer): The optimizer to use for the supervised (q) model learning task.

            num_steps_per_update_q (int): The number of full batch gradient descent iterations on (q) per update
                (each iteration uses the same environment samples).

            num_steps_per_update_pi (int): The number of full batch gradient descent iterations on (pi) per update
                (each iteration uses the same environment samples).

            minibatch_size_q (int): Full batch from buffer will be broken up into minibatches of this size

            minibatch_size_pi (int): Full batch from buffer will be broken up into minibatches of this size

            episode_buffer_capacity (int): The capacity of the episode (experience) FIFOBuffer.

            max_time_steps (Optional[int]): The maximum number of time steps (across all actors) to learn/update
                the dynamics-(q)-model. If None, use a value given by the environment.

            summaries (List[any]): A list of summaries to produce if `UseTfSummaries` in debug.json is true.
                In the simplest case, this is a list of `self.[...]`-property names of the SAC object that should
                be tracked after each tick.
        """
        # Clean up network configs to be passable as **kwargs to `make`.
        # Networks are given as sequential config or directly as Keras objects -> prepend "network" key to spec.
        if isinstance(policy_network, (list, tuple, tf.keras.models.Model, tf.keras.layers.Layer)):
            policy_network = dict(network=policy_network)
        if isinstance(q_network, (list, tuple, tf.keras.models.Model, tf.keras.layers.Layer)):
            q_network = dict(network=q_network)

        # Make state/action space.
        state_space = Space.make(state_space)
        action_space = Space.make(action_space)
        skill_space = Float(-1.0, 1.0, shape=(dim_skill_vectors,), main_axes="B") if \
            discrete_skills is False else Int(dim_skill_vectors, main_axes="B")

        sac_config = SACConfig.make(
            sac_config,
            state_space=Dict(s=state_space, z=skill_space),
            action_space=action_space,
            # Use no memory. Updates are done from DADS' own buffer.
            memory_capacity=1, memory_batch_size=1,
            # Share policy network between DADS and underlying learning SAC.
            policy_network=policy_network
        )

        if skill_horizon is None:
            skill_horizon = episode_horizon

        super().__init__(locals())  # Config will store all c'tor variables automatically.

        # Keep track of which time-step stuff happened. Only important for by-time-step frequencies.
        self.last_update = 0
