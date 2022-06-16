from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import numpy as np
import torch
from absl import app
from absl import flags
import collections
import tensorflow.compat.v1 as tf
import pyspiel

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python import rl_environment
from open_spiel.python.rl_environment import TimeStep, StepType
import mb_deep_cfr_oracle as deep_cfr
from network.env_model import DynamicModel
import os.path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 3, "Seed.")

# game_setting
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("n_players", 5, "Number of players")
flags.DEFINE_integer("game_length", 10, "Game Length")
flags.DEFINE_integer("iterations", 200, "Number of training iterations.")

# algorithm setting
flags.DEFINE_integer("num_traversals", 100, "Number of traversals/games")
flags.DEFINE_integer("batch_size_advantage", 256, "Adv fn batch size")
flags.DEFINE_integer("batch_size_strategy", 64, "Strategy batch size")
flags.DEFINE_integer("num_hidden", 64, "Hidden units in each layer")
flags.DEFINE_integer("num_layers", 3, "Depth of neural networks")
flags.DEFINE_bool("reinitialize_advantage_networks", False, "Re-init value net on each CFR iter")
flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate")
flags.DEFINE_integer("memory_capacity", 10000000, "replay buffer capacity")
flags.DEFINE_integer("policy_network_train_steps", 200, "training steps per iter")
flags.DEFINE_integer("advantage_network_train_steps", 100, "training steps per iter")

flags.DEFINE_bool("use_round", True, "env model whether use round to normalize state")
flags.DEFINE_bool("use_deterministic", True, "Re-init value net on each CFR iter")
flags.DEFINE_string("env_model_location", "mix_offline_dataset_trained_env_model/", "location of env model")
flags.DEFINE_string("env_model_file", "game_kuhn_poker_players_5_hidden_layer_64_buffer_{}_lr_0.05_train_epoch_{}_batch_size_128.pkl", "location of env model")
flags.DEFINE_integer("replay_buffer", 20000, "env model replay buffer")
flags.DEFINE_integer("train_epoch", 5000, "env model replay buffer")

flags.DEFINE_string("device", "cuda", "device")
FLAGS.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_policy_result_dir(proportion):
    result_dir = "mb_method_results/mb_deep_cfr_train_policy/" + FLAGS.game_name + '_' + str(FLAGS.n_players) + "_players/train_data_{}".format(FLAGS.replay_buffer)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    if not FLAGS.use_round:
        result_name = "seed_{}".format(FLAGS.seed) + "_not_use_round_policy_train_epoch_" + str(FLAGS.train_epoch) + \
                      "_proportion_" + str(proportion) + ".pkl"
    else:
        result_name = "seed_{}".format(FLAGS.seed) + "_policy_train_epoch_" + str(FLAGS.train_epoch) + "_proportion_" + str(proportion) + ".pkl"
    return osp.join(result_dir, result_name)


def get_nash_conv_result_dir():
    result_dir = "mb_method_results/mb_deep_cfr_train_nash_conv/" + FLAGS.game_name + '_' + str(FLAGS.n_players) + "_players/train_data_{}".format(FLAGS.replay_buffer)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    if not FLAGS.use_round:
        result_name = "seed_{}".format(FLAGS.seed) + "_not_use_round_train_epoch_" + str(FLAGS.train_epoch) + ".pkl"
    else:
        result_name = "seed_{}".format(FLAGS.seed) + "_train_epoch_" + str(FLAGS.train_epoch) + ".pkl"
    return osp.join(result_dir, result_name)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_random_seed(seed)
    torch.backends.cudnn.deterministic = True


def reset_state(game, env_model):
    state = game.new_initial_state()
    chance_action = len(state.chance_outcomes())

    next_state = [state.information_state_tensor(player_id) for player_id in range(FLAGS.n_players)]
    legal_action = [i for i in range(chance_action)]
    observations = {"info_state": next_state, "legal_actions": [legal_action for _ in range(FLAGS.n_players)],
                    "current_player": -1, "serialized_state": []}
    step_type = StepType.FIRST
    rewards = [0, 0]

    action = np.random.choice([i[0] for i in state.chance_outcomes()])
    time_step = TimeStep(observations=observations, rewards=rewards, discounts=0.0, step_type=step_type)

    return env_model.step(time_step, [action])


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # set seed
    setup_seed(FLAGS.seed)

    # load liar's dice game
    # game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players, "numdice": FLAGS.numdice, "dice_sides": 6})
    # load poker game
    game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players})
    # load phantom ttt game
    # game = pyspiel.load_game(FLAGS.game_name, {"obstype": "reveal-nothing"})
    state = game.new_initial_state()
    chance_action = len(state.chance_outcomes())
    num_actions = max(game.num_distinct_actions(), chance_action)

    env = rl_environment.Environment(game)
    env.seed(seed=FLAGS.seed)
    nash_conv_list = []

    # load env model
    for proportion in range(11):
        print("Start Run Proportion:", proportion)
        model_location = FLAGS.env_model_location + FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players/" + "env_model_proportion_" \
                         + str(proportion) + "/" + FLAGS.env_model_file.format(FLAGS.replay_buffer, FLAGS.train_epoch)

        trained_model = torch.load(model_location)
        env_args = {"state_length": env.observation_spec()["info_state"][0],
                    "env_action_number": num_actions,
                    "legal_action_number": game.num_distinct_actions(),
                    "player_number": FLAGS.n_players,
                    "game_length": FLAGS.game_length,
                    "env_model": trained_model,
                    "use_round": FLAGS.use_round,
                    "use_deterministic_env_model": FLAGS.use_deterministic,
                    "device": FLAGS.device}

        # define env model class
        env_model = DynamicModel(**env_args)

        deep_cfr_solver = deep_cfr.DeepCFRSolver(
            game,
            env_model,
            device=FLAGS.device,
            policy_network_layers=tuple([FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
            advantage_network_layers=tuple([FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
            num_iterations=FLAGS.iterations,
            num_traversals=FLAGS.num_traversals,
            learning_rate=FLAGS.learning_rate,
            batch_size_advantage=FLAGS.batch_size_advantage,
            batch_size_strategy=FLAGS.batch_size_strategy,
            memory_capacity=FLAGS.memory_capacity,
            policy_network_train_steps=FLAGS.policy_network_train_steps,
            advantage_network_train_steps=FLAGS.advantage_network_train_steps,
            reinitialize_advantage_networks=FLAGS.reinitialize_advantage_networks)

        # use deep cfr to solve the game

        """Solution logic for Deep CFR."""
        advantage_losses = collections.defaultdict(list)
        for i in range(deep_cfr_solver._num_iterations):
            print("Iteration:", i)
            for p in range(deep_cfr_solver._num_players):
                for _ in range(deep_cfr_solver._num_traversals):
                    deep_cfr_solver.mb_traverse_game_tree(reset_state(game, env_model), p, FLAGS.game_length)

                if deep_cfr_solver._reinitialize_advantage_networks:
                    deep_cfr_solver.reinitialize_advantage_network(p)
                advantage_losses[p].append(deep_cfr_solver._learn_advantage_network(p))

            deep_cfr_solver._iteration += 1

        deep_cfr_solver._learn_strategy_network()
        # compute the final nash conv
        average_policy = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
        conv = exploitability.nash_conv(game, average_policy)
        nash_conv_list.append(conv)
        best_policy = copy.deepcopy(deep_cfr_solver.get_policy_network())
        print("conv:", nash_conv_list)
        torch.save(best_policy, get_policy_result_dir(proportion))
        torch.save(nash_conv_list, get_nash_conv_result_dir())


if __name__ == "__main__":
    app.run(main)
