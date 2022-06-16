import sys
import os
import torch
import random
import pyspiel
import numpy as np
from absl import app
import os.path as osp
from absl import flags
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from policy_wrapper import deep_cfr_policy
from network.ensemble_mb_bc import mix_policy


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 1, "Seed.")
flags.DEFINE_integer("proportion", 10, "dataset proportion")

flags.DEFINE_string("game_name", "kuhn_poker", "Game name.")
flags.DEFINE_integer("n_players", 4, "The number of players.")

# Behavior Clone Strategy Location
flags.DEFINE_string("bc_policy_location", "mix_offline_dataset_behavior_clone_policy/", "offline data location")
flags.DEFINE_string("bc_policy_file_name",
                    "/seed_1_game_kuhn_poker_players_4_hidden_layer_64_buffer_10000_lr_0.05_train_epoch_5000_batch_size_128_policy.pkl",
                    "Behavior Clone Strategy Location")

flags.DEFINE_string("mb_policy_location", "mb_deep_cfr/mb_method_results/mb_deep_cfr_train_policy/", "offline data location")
flags.DEFINE_string("mb_policy_file_name",
                    "/policy_train_data_10000_train_epoch_5000_proportion_10_conv_0.6931734493672149.pkl",
                    "Behavior Clone Strategy Location")

flags.DEFINE_integer("replay_buffer", 500, "env model replay buffer")

flags.DEFINE_string("device", "cpu", "device type")
FLAGS.device = "cuda" if torch.cuda.is_available() else "cpu"


def tabular_policy_from_callable(game, behavior_policy, players=None):
    tabular_policy = policy.TabularPolicy(game, players)
    for state_index, state in enumerate(tabular_policy.states):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        info_state_vector = np.array(state.information_state_tensor())
        if len(info_state_vector.shape) == 1:
            info_state_vector = np.expand_dims(info_state_vector, axis=0)
        info_state_vector = torch.FloatTensor(info_state_vector).to(FLAGS.device)

        strategy = behavior_policy[cur_player].step(info_state_vector).squeeze(0).tolist()

        action_probabilities = {action: strategy[action] for action in legal_actions}

        infostate_policy = [action_probabilities.get(action, 0.) for action in range(game.num_distinct_actions())]
        tabular_policy.action_probability_array[state_index, :] = infostate_policy
    return tabular_policy


def get_result_dir():
    result_dir = "results_bc/" + FLAGS.game_name + '_' + str(FLAGS.n_players) + "_players"
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    result_name = "game_{}_players_{}_replay_buffer_{}.txt".format(FLAGS.game_name, FLAGS.n_players, FLAGS.replay_buffer)
    return osp.join(result_dir, result_name)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # set seed
    setup_seed(FLAGS.seed)

    # load game
    game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players})
    results = []
    min_weights = []
    for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        conv_list = []
        policy_list = []

        for index in range(FLAGS.n_players):
            bc_location = FLAGS.bc_policy_location + FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players/" + \
                       str(index) + "_player_policy_proportion_" + str(FLAGS.proportion) + FLAGS.bc_policy_file_name
            mb_location = FLAGS.mb_policy_location + FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players" + FLAGS.mb_policy_file_name
            deep_cfr_bc_model = deep_cfr_policy(torch.load(mb_location).to(FLAGS.device), device=FLAGS.device)

            policy_list.append(mix_policy(bc_model=torch.load(bc_location).to(FLAGS.device), mb_model=deep_cfr_bc_model, bc_weight=w))

        # compute nash_cov
        average_policy = tabular_policy_from_callable(game, policy_list)
        conv = exploitability.nash_conv(game, average_policy)
        conv_list.append(conv)

        results.append(min(conv_list))
        min_weights.append(weights_list[conv_list.index(min(conv_list))])
        print(conv)

    print(results)
    torch.save([results, min_weights], get_result_dir())


if __name__ == "__main__":
    app.run(main)
