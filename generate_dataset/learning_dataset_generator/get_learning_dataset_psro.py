import os
import time
import copy
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

import torch_rl_policy_record_dataset as rl_policy
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 1, "Seed.")
flags.DEFINE_string("game_name", "kuhn_poker", "Game name.")
flags.DEFINE_integer("n_players", 3, "The number of players.")
flags.DEFINE_integer("gpsro_iterations", 100, "Number of training steps for GPSRO.")

# PSRO related
flags.DEFINE_integer("number_training_episodes", int(1e4), "Number training episodes per RL policy.")
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

flags.DEFINE_string("device", "cuda", "device")
FLAGS.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_result_dir(iteration, nashcon):
    result_dir = "expert_policy/psro"
    sub_folder = FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players"
    result_dir = osp.join(result_dir, sub_folder)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    result_name = "train_{}_iterations_nash_conv_{}.pth".format(iteration, nashcon)
    return osp.join(result_dir, result_name)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)
    torch.backends.cudnn.deterministic = True


# initialize DQN agent
def init_dqn_responder(env):
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

    # define psro solver
    g_psro_solver = psro_v2.PSROSolver(env.game,
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

    # run psro iterations
    start_time = time.time()
    best_exploi = 100
    for gpsro_iteration in range(FLAGS.gpsro_iterations):
        print("Iteration : {}".format(gpsro_iteration))
        print("Time so far: {}".format(time.time() - start_time))

        g_psro_solver.iteration()

        meta_probabilities = g_psro_solver.get_meta_strategies()
        policies = g_psro_solver.get_policies()

        # compute exploitability for policies
        aggregator = policy_aggregator.PolicyAggregator(env.game)
        aggr_policies = aggregator.aggregate(range(FLAGS.n_players), policies, meta_probabilities)
        exploitabilities, _ = exploitability.nash_conv(env.game, aggr_policies, return_only_nash_conv=False)

        if best_exploi > exploitabilities:
            # save policy
            best_exploi = exploitabilities
            policy_list = [[p[i]._policy for i in range(len(p))] for p in policies]
            torch.save([policy_list, copy.deepcopy(meta_probabilities)], get_result_dir(gpsro_iteration + 1, exploitabilities))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    setup_seed(FLAGS.seed)

    # load game
    game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players})
    env = rl_environment.Environment(game)
    env.seed(FLAGS.seed)

    # Initialize oracle and agents
    oracle, agents = init_dqn_responder(env)
    gpsro_looper(env, oracle, agents)


if __name__ == "__main__":
    app.run(main)
