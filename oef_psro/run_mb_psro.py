import os
import time
import torch
import random
import pyspiel
import numpy as np
from absl import app
import os.path as osp
from absl import flags
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
import mb_psro as psro_v2
import mb_psro_rl_oracle as rl_oracle
import torch_rl_policy as rl_policy
from network.env_model import DynamicModel

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 2, "Seed.")
# flags.DEFINE_integer("proportion", 10, "dataset proportion")

flags.DEFINE_string("game_name", "leduc_poker", "Game name.")
flags.DEFINE_integer("n_players", 3, "The number of players.")
flags.DEFINE_integer("game_length", 10, "Game Length")
flags.DEFINE_integer("gpsro_iterations", 30, "Number of training steps for GPSRO.")

# PSRO related
flags.DEFINE_integer("number_training_episodes", int(1e3), "Number training episodes per RL policy.")
flags.DEFINE_string("meta_strategy_method", "alpharank", "Name of meta strategy computation method.")
flags.DEFINE_integer("number_policies_selected", 1, "Number of new strategies trained at each PSRO iteration.")
flags.DEFINE_integer("sims_per_entry", 1000, "Number of simulations to estimate elements of the game outcome matrix.")
flags.DEFINE_integer("prd_iterations", 50000, "Number of training steps for PRD.")
flags.DEFINE_bool("symmetric_game", False, "Whether to consider the current game as a symmetric game.")

# Rectify options
flags.DEFINE_string("rectifier", "", "Which rectifier to use. Choices are ''(No filtering),'rectified' for rectified.")
flags.DEFINE_string("training_strategy_selector", "probabilistic",
                    "Which strategy selector to use. Choices are "
                    " - 'top_k_probabilities': select top `number_policies_selected` strategies. "
                    " - 'probabilistic': Randomly samples `number_policies_selected` strategies with probability "
                    "equal to their selection probabilities. "
                    " - 'uniform': Uniformly sample `number_policies_selected` strategies. "
                    " - 'rectified': Select every non-zero-selection-probability strategy available to each player.")

# General (RL) agent parameters
flags.DEFINE_integer("hidden_layer_size", 256, "Hidden layer size")
flags.DEFINE_integer("n_hidden_layers", 4, "# of hidden layers")
flags.DEFINE_float("dqn_learning_rate", 1e-2, "DQN learning rate.")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_float("sigma", 0.0, "Policy copy noise (Gaussian Dropout term).")
flags.DEFINE_string("optimizer_str", "adam", "'adam' or 'sgd'")
flags.DEFINE_integer("learn_every", 10, "Learn every [X] steps.")
flags.DEFINE_float("self_play_proportion", 0.0, "Self play proportion")
flags.DEFINE_integer("update_target_network_every", 1000, "Update target network every [X] steps")

flags.DEFINE_bool("use_round", True, "env model whether use round to normalize state")
flags.DEFINE_bool("use_deterministic", True, "Re-init value net on each CFR iter")
flags.DEFINE_string("env_model_location", "mix_offline_dataset_trained_env_model/", "location of env model")
flags.DEFINE_string("env_model_file", "game_leduc_poker_players_3_hidden_layer_128_buffer_{}_lr_0.05_train_epoch_{}_batch_size_128.pkl", "location of env model")
flags.DEFINE_integer("replay_buffer", 50000, "env model replay buffer")
flags.DEFINE_integer("train_epoch", 10000, "env model replay buffer")

flags.DEFINE_string("device", "cuda", "device")
FLAGS.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_policy_result_dir(proportion):
    result_dir = "mb_method_results/mb_psro_train_policy/" + FLAGS.game_name + '_' + str(FLAGS.n_players) + "_players/train_data_{}".format(FLAGS.replay_buffer)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    if not FLAGS.use_round:
        result_name = "seed_{}".format(FLAGS.seed) + "_not_use_round_policy_env_train_epoch_" + str(FLAGS.train_epoch) + "_number_training_episodes_" + str(FLAGS.number_training_episodes) + \
                      "_proportion_" + str(proportion) + ".pkl"
    else:
        result_name = "seed_{}".format(FLAGS.seed) + "_policy_env_train_epoch_" + str(FLAGS.train_epoch) + "_number_training_episodes_" + str(FLAGS.number_training_episodes) +"_proportion_" + str(proportion) + ".pkl"
    return osp.join(result_dir, result_name)


def get_nash_conv_result_dir():
    result_dir = "mb_method_results/mb_psro_train_nash_conv/" + FLAGS.game_name + '_' + str(FLAGS.n_players) + "_players/train_data_{}".format(FLAGS.replay_buffer)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    if not FLAGS.use_round:
        result_name = "seed_{}".format(FLAGS.seed) + "_not_use_round_train_epoch_" + str(FLAGS.train_epoch) + "_number_training_episodes_" + str(FLAGS.number_training_episodes) + ".pkl"
    else:
        result_name = "seed_{}".format(FLAGS.seed) + "_train_epoch_" + str(FLAGS.train_epoch) + "_number_training_episodes_" + str(FLAGS.number_training_episodes) + ".pkl"
    return osp.join(result_dir, result_name)


def init_dqn_responder(env, env_model):
    state_representation_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent_class = rl_policy.DQNPolicy
    agent_kwargs = {
        "state_representation_size": state_representation_size,
        "num_actions": num_actions,
        "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
        "batch_size": FLAGS.batch_size,
        "learning_rate": FLAGS.dqn_learning_rate,
        "update_target_network_every": FLAGS.update_target_network_every,
        "learn_every": FLAGS.learn_every,
        "optimizer_str": FLAGS.optimizer_str
    }
    oracle = rl_oracle.RLOracle(
        env,
        env_model,
        FLAGS.game_length,
        agent_class,
        agent_kwargs,
        number_training_episodes=FLAGS.number_training_episodes,
        self_play_proportion=FLAGS.self_play_proportion,
        sigma=FLAGS.sigma)

    agents = [agent_class(env, player_id, **agent_kwargs) for player_id in range(FLAGS.n_players)]

    for agent in agents:
        agent.freeze()
    return oracle, agents


def gpsro_looper(env, oracle, agents):
    """Initializes and executes the GPSRO training loop."""
    print("Game : {}".format(FLAGS.game_name))
    print("Seed: {}".format(FLAGS.seed))

    sample_from_marginals = True
    training_strategy_selector = FLAGS.training_strategy_selector or strategy_selectors.probabilistic_strategy_selector

    g_psro_solver = psro_v2.PSROSolver(
        env.game,
        oracle,
        initial_policies=agents,
        training_strategy_selector=training_strategy_selector,
        rectifier=FLAGS.rectifier,
        sims_per_entry=FLAGS.sims_per_entry,
        number_policies_selected=FLAGS.number_policies_selected,
        meta_strategy_method=FLAGS.meta_strategy_method,
        prd_iterations=FLAGS.prd_iterations,
        prd_gamma=1e-10,
        sample_from_marginals=sample_from_marginals,
        symmetric_game=FLAGS.symmetric_game)

    start_time = time.time()
    for gpsro_iteration in range(FLAGS.gpsro_iterations):
        print("Iteration : {}".format(gpsro_iteration))
        print("Time so far: {}".format(time.time() - start_time))

        g_psro_solver.iteration()

    return g_psro_solver


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    setup_seed(FLAGS.seed)

    game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players})
    state = game.new_initial_state()
    chance_action = len(state.chance_outcomes())
    num_actions = max(game.num_distinct_actions(), chance_action)

    env = rl_environment.Environment(game)
    env.seed(FLAGS.seed)
    ex_list = []
    for proportion in range(7, 11):
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

        env_model = DynamicModel(**env_args)

        # Initialize oracle and agents
        oracle, agents = init_dqn_responder(env, env_model)
        g_psro_solver = gpsro_looper(env, oracle, agents)
        # evaluate
        meta_probabilities = g_psro_solver.get_meta_strategies()
        policies = g_psro_solver.get_policies()
        # compute nash conv
        aggregator = policy_aggregator.PolicyAggregator(env.game)
        aggr_policies = aggregator.aggregate(range(FLAGS.n_players), policies, meta_probabilities)
        exploitabilities, expl_per_player = exploitability.nash_conv(env.game, aggr_policies,
                                                                     return_only_nash_conv=False)
        # save exploitability
        ex_list.append(exploitabilities)
        best_policy = [[p[i]._policy for i in range(len(p))] for p in policies]
        best_policy.append(meta_probabilities)

        print("conv:", ex_list)
        torch.save(ex_list, get_nash_conv_result_dir())
        torch.save(best_policy, get_policy_result_dir(proportion))


if __name__ == "__main__":
    app.run(main)
