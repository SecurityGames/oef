from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import random
import pyspiel
import numpy as np
from absl import app
import os.path as osp
from absl import flags
import tensorflow.compat.v1 as tf
from torch_deep_cfr_solver_record_dataset import DeepCFRSolver

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 1, "Seed.")
flags.DEFINE_integer("n_players", 3, "Number of players")
flags.DEFINE_string("game_name", "leduc_poker", "Name of the game")
flags.DEFINE_integer("iterations", 10, "Number of training iterations.")

# algorithm setting
flags.DEFINE_integer("num_traversals", 200, "Number of traversals/games")
flags.DEFINE_integer("batch_size_advantage", 256, "Adv fn batch size")
flags.DEFINE_integer("batch_size_strategy", 64, "Strategy batch size")
flags.DEFINE_integer("num_hidden", 64, "Hidden units in each layer")
flags.DEFINE_integer("num_layers", 3, "Depth of neural networks")
flags.DEFINE_bool("reinitialize_advantage_networks", False, "Re-init value net on each CFR iter")
flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate")
flags.DEFINE_integer("memory_capacity", 10000000, "replay buffer capacity")
flags.DEFINE_integer("policy_network_train_steps", 200, "training steps per iter")
flags.DEFINE_integer("advantage_network_train_steps", 100, "training steps per iter")
flags.DEFINE_string("device", "cuda", "device")
FLAGS.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_result_dir(data_size, conv):
    result_dir = "learning_dataset/deep_cfr"
    sub_folder = FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players"
    result_dir = osp.join(result_dir, sub_folder)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    buffer_name = "dataset_size_{}_train_{}_iterations_best_nash_conv_{}.pth".format(data_size, FLAGS.iterations, conv)
    return osp.join(result_dir, buffer_name)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_random_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # set seed
    setup_seed(FLAGS.seed)
    game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players})

    deep_cfr_solver = DeepCFRSolver(game,
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
    buffer, conv = deep_cfr_solver.solve()
    # compute the best_model nash conv
    torch.save(buffer, get_result_dir(len(buffer), conv))


if __name__ == "__main__":
    app.run(main)
