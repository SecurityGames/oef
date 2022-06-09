# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
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

"""Example use of the CFR algorithm on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import copy
import torch
from absl import app
from absl import flags
import os.path as osp

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "sampling",
    "external",
    ["external", "outcome"],
    "Sampling for the MCCFR solver",
)
flags.DEFINE_integer("iterations", int(2e6), "Number of iterations")
# flags.DEFINE_string("game_name", "liars_dice", "Name of the game")
# flags.DEFINE_string("game_name", "leduc_poker", "Name of the game")
flags.DEFINE_string("game_name", "phantom_ttt", "Name of the game")
flags.DEFINE_integer("n_players", 2, "Number of players")
flags.DEFINE_integer("numdice", 2, "Number of players")

flags.DEFINE_integer("print_freq", int(1e6), "How often to print the exploitability")


def get_result_dir(iterations, conv):
    result_dir = "expert_policy/cfr"

    if FLAGS.game_name == "liars_dice":
        sub_folder = FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players_" + str(FLAGS.numdice) + "_numdice"
    else:
        sub_folder = FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players"

    result_dir = osp.join(result_dir, sub_folder)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    buffer_name = "train_{}_iterations_nash_conv_{}.pth".format(iterations, conv)
    return osp.join(result_dir, buffer_name)


def main(_):
    game = pyspiel.load_game(FLAGS.game_name, {"obstype": "reveal-numturns"})

    # game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players})
    # game = pyspiel.load_game(FLAGS.game_name, {"dice_sides": 6, "numdice": FLAGS.numdice, "players": FLAGS.n_players})
    # print("s")
    # cfr_solver = cfr.CFRSolver(game)
    # best_conv = 100
    # print("ok")
    #
    # for i in range(FLAGS.iterations):
    #     print(i)
    #     cfr_solver.evaluate_and_update_policy()
    #     if (i + 1) % FLAGS.print_freq == 0:
    #         conv = exploitability.nash_conv(game, cfr_solver.average_policy())
    #         print("Iteration {} exploitability {}".format(i, conv))
    #         if conv < best_conv:
    #             best_conv = conv
    #             best_model = copy.deepcopy(cfr_solver.average_policy())
    #             torch.save(best_model, get_result_dir(i + 1, best_conv))
    #             if best_conv <= 0.05:
    #                 break

    # game = pyspiel.load_game(FLAGS.game_name, {"dice_sides": 6, "numdice": FLAGS.numdice, "players": FLAGS.n_players})
    # game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players})
    if FLAGS.sampling == "external":
        cfr_solver = external_mccfr.ExternalSamplingSolver(
            game, external_mccfr.AverageType.SIMPLE)
    else:
        cfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)

    print("compute exploitability......")
    conv = exploitability.nash_conv(game, cfr_solver.average_policy(), return_only_nash_conv=False, use_cpp_br=False)
    print(conv)

    best_conv = 100
    for i in range(FLAGS.iterations):
        # print(i)
        cfr_solver.iteration()
        if (i + 1) % 10000 == 0:
            print(i + 1)
        if (i + 1) % FLAGS.print_freq == 0:
            print("compute exploitability......")
            conv = exploitability.nash_conv(game, cfr_solver.average_policy(), return_only_nash_conv=False, use_cpp_br=False)
            print("Iteration {} exploitability {}".format(i, conv))
            if conv < best_conv:
                best_conv = conv
                best_model = copy.deepcopy(cfr_solver.average_policy())
                torch.save(best_model, get_result_dir(i + 1, best_conv))
                if best_conv < 0.05:
                    break


if __name__ == "__main__":
    app.run(main)
