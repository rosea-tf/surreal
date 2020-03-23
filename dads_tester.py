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
import json

from surreal.algos.dads import DADS, DADSConfig
from surreal.components import Preprocessor
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv, GridWorld
from surreal import SURREAL_HOME, parser


parser.add_argument('--config', default="surreal/tests/algos/configs/dads_grid_world_4room_learning.json")
parser.add_argument('--action_type', default="udlr")

args, _ = parser.parse_known_args()

logging.getLogger().setLevel(logging.INFO)

def test_dads_learning_on_grid_world_4room():
    # Create an Env object.
    env = GridWorld(
        "4-room",
        state_representation="xy",
        action_type=args.action_type
    )

    # Add the preprocessor.
    preprocessor = Preprocessor(
        lambda inputs_: tf.reshape(
            tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        , shape=(1,env.actors[0].state_space.flat_dim_with_categories))
    )

    # add path tracker
    class DADS_Path_Tracker():

        def init(self, algo, config):
            self.curr_path = None
            self.curr_z = None
            self.path_by_skill = {
                k: []
                for k in range(algo.config.dim_skill_vectors)
            }

        def track(self, algo, event):
            self.curr_path.append(tuple([int(i) for i in event.s_[0]]))
        
        def collect(self, algo, event):
            if self.curr_z is not None:
                self.path_by_skill[self.curr_z].append(self.curr_path)

            self.curr_z = algo.z.value[0]
            self.curr_path = []

    dads_path_tracker = DADS_Path_Tracker()


    # Create a Config.
    config = DADSConfig.make(
        args.config,
        preprocessor=preprocessor,
        state_space=env.actors[0].state_space,
        action_space=env.actors[0].action_space,
        summaries=[
            "q_loss",
            "ri",  #intrinsic rewards
            "skill_divergence",
            "SAC.L_actor",
            "SAC.L_alpha",
            "SAC.Ls_critic[0]",
            "SAC.alpha",
        ],
        callbacks={
            "__init__": [dads_path_tracker.init],
            "event_tick": [dads_path_tracker.track],
            "event_episode_starts": [dads_path_tracker.collect],
        },
    )

    # Create an Algo object.
    algo = DADS(config=config, name="my-dads")

    # Point actor(s) to the algo.
    env.point_all_actors_to_algo(algo)

    # Run and wait for env to complete.
    env.run(ticks=300000, sync=True, render=debug.RenderEnvInLearningTests, max_episode_length=20)

    env.terminate()

    algo.save_weights(os.path.join(SURREAL_HOME, "weights.gz"))

    with open(os.path.join(SURREAL_HOME, "path_by_skill.json"), "w") as write_file:
        json.dump(dads_path_tracker.path_by_skill, write_file)


if __name__ == '__main__':
    test_dads_learning_on_grid_world_4room()
