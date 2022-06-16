import os
import torch
import random
import pyspiel
import numpy as np
from absl import app
import os.path as osp
from absl import flags
from utils import policy_wrapper
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 1, "Seed.")
flags.DEFINE_integer("n_players", 3, "Number of players")

# flags.DEFINE_string("game_name", "liars_dice", "Name of the game")
flags.DEFINE_integer("numdice", 1, "Number of players")

flags.DEFINE_string("game_name", "leduc_poker", "Name of the game")
# flags.DEFINE_string("game_name", "phantom_ttt", "Name of the game")

flags.DEFINE_integer("num_episode", int(1e4), "the number of sample episodes")

# policy related
flags.DEFINE_string("data_type", "random_data", "Type of data: random_data, expert_data_deep_cfr, expert_data_psro")
flags.DEFINE_string("result_folder", "random_dataset", "Type of data: random_dataset, expert_dataset")

flags.DEFINE_string("device", "cuda", "device")
FLAGS.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_result_dir():
    if FLAGS.game_name == "liars_dice":
        result_dir = FLAGS.result_folder + "/" + FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players_" + str(FLAGS.numdice) + "_numdice"
    else:
        result_dir = FLAGS.result_folder + "/" + FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players"
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    buffer_name = "seed_{}_{}_players_{}_episode_{}".format(FLAGS.seed, FLAGS.game_name, FLAGS.n_players, FLAGS.num_episode)
    buffer_name += ".pth"
    return osp.join(result_dir, buffer_name)


def sample_save_data(game, num_actions, policy):
    buffer = []
    for epi in range(FLAGS.num_episode):
        state = game.new_initial_state()

        while not state.is_terminal():
            # current information state list and current player id
            pre_info_state = [state.information_state_tensor(player_id) for player_id in range(FLAGS.n_players)]
            player_id = state.current_player()

            # chance node
            if state.is_chance_node():
                legal_actions = [i[0] for i in state.chance_outcomes()]
                action = np.random.choice([i[0] for i in state.chance_outcomes()])
            else:
                legal_actions = state.legal_actions()
                strategy = policy[player_id](state) if FLAGS.data_type == "expert_data_psro" else policy(state)
                action = np.random.choice(range(num_actions), p=strategy)

            # get next state
            state = state.child(action)

            next_info_state = [state.information_state_tensor(player_id) for player_id in range(FLAGS.n_players)]
            next_legal_actions = [i[0] for i in state.chance_outcomes()] if state.is_chance_node() else state.legal_actions()
            next_player = state.current_player()
            reward = state.returns()

            done = [1] if state.is_terminal() else [0]
            chance_node = [1] if state.is_chance_node() else [0]

            # current_info_state, player_id, legal_actions, action, next_info_state, next_legal_actions,
            # next_player, reward, done, chance_node
            transition = [pre_info_state, player_id, legal_actions, [action], next_info_state,
                          next_legal_actions, next_player, reward, done, chance_node]
            buffer.append(transition)

    torch.save(buffer, get_result_dir())


# set seed
def set_seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    set_seed_all(FLAGS.seed)

    game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players})
    # game = pyspiel.load_game(FLAGS.game_name, {"obstype": "reveal-nothing"})
    # game = pyspiel.load_game(FLAGS.game_name, {"dice_sides": 6, "numdice": FLAGS.numdice, "players": FLAGS.n_players})

    num_actions = game.num_distinct_actions()
    
    policy = policy_wrapper.random_policy(num_actions)
    sample_save_data(game, num_actions, policy)


if __name__ == "__main__":
    app.run(main)
