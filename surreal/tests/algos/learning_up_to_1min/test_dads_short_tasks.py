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

import logging
import numpy as np
import os
import tensorflow as tf
import unittest

from surreal.algos.dads import DADS, DADSConfig
from surreal.components import Preprocessor
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv, GridWorld
from surreal.tests.test_util import check


class TestDADSShortLearningTasks(unittest.TestCase):
    """
    Tests the DADS algo on shorter-than-1min learning problems.
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_dads_learning_on_grid_world_4room(self):
        # Create an Env object.
        env = GridWorld("4-room")

        # Add the preprocessor.
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        )

        # add a reward function for gridworld
        # @staticmethod
        def reward_function(state):
            """
            state given as (b,121) array
            returns: reward, terminal
            """
            state_ = np.argmax(state, axis=-1)
            state_xy = np.stack(env.processes[0][0]._get_x_y(state_), axis=-1)[:, None, :] # n_actors x 1 x 2
            goal_xy = np.stack([env.processes[0][0].goal_x, env.processes[0][0].goal_y], axis=-1) # n_goals x 2
            min_manhattan_dist = np.min(
                np.sum(np.abs(goal_xy - state_xy), axis=2) # n_actors x n_goals
                , axis=1) # n_actors
            return -1 * min_manhattan_dist, min_manhattan_dist == 0

        # Create a Config.
        config = DADSConfig.make(
            "{}/../configs/dads_grid_world_4room_learning.json".format(os.path.dirname(__file__)),
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space,
            reward_function=reward_function,
            plan_horizon=2, #TODO
            summaries=[
                "q_loss",
                "SAC.L_actor",
                "SAC.L_alpha",
                "SAC.Ls_critic[0]",
                "avg_ri", #intrinsic rewards
                "skill_divergence",
                "skill_uniqueness"
            ]
        )

        # Create an Algo object.
        algo = DADS(config=config, name="my-dads")
        # algo.inference = True

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=3000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        # TODO this is all just copied from DQN2015!
        # n = 10
        # mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        # print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        # self.assertTrue(mean_last_n >= 0.3)

        # Check learnt Q-function.
        # check(algo.q(
        #     np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        # ), [[0.8, -5.0, 0.9, 0.8], [0.8, 1.0, 0.9, 0.9]], decimals=1)  # a=up,down,left,right

        env.terminate()
