import torch
# import sys
# sys.path.append("/home/kangjie/lsx/oef/oef_kuhn_2")
import random
import pyspiel
import numpy as np
from absl import app
import os.path as osp
from absl import flags
import torch.utils.data as Data
from network.env_model import EnvModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

FLAGS = flags.FLAGS

# Game-related
flags.DEFINE_string("game_name", "liars_dice", "Game name.")
flags.DEFINE_integer("n_players", 2, "The number of players.")
flags.DEFINE_integer("numdice", 1, "The number of players.")

# General
flags.DEFINE_integer("seed", 1, "Seed.")

# offline setting
flags.DEFINE_string("mix_offline_data_location",
                    "/home/kangjie/lsx/oef/oef_liars_dice/dataset/mix_offline_dataset",
                    "offline data location")
# record results location
flags.DEFINE_string("result_data_location", "mix_offline_dataset_trained_env_model", "dataset class")
flags.DEFINE_float("exploitability", 0.03779695063020872, "the location of trained expert policy")

# train env model
flags.DEFINE_float("learning_rate", 5e-2, "train env model learning rate.")
flags.DEFINE_integer("hidden_layer_size", 64, "env model Hidden layer size")
flags.DEFINE_integer("data_number", 50000, "The number of data.")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("train_epoch", 5000, "batch size")
# flags.DEFINE_integer("save_every", int(1e3), "save model every")

flags.DEFINE_string("device", "cpu", "device type")
FLAGS.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_result_dir(epoch, proportion):
    result_dir = str(FLAGS.result_data_location) + "/" + FLAGS.game_name + '_' + str(FLAGS.n_players) + "_players_" + str(FLAGS.numdice) + "_numdice"
    sub_folder = "env_model_proportion_" + str(proportion)
    result_dir = osp.join(result_dir, sub_folder)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    model_name = "game_{}_players_{}_hidden_layer_{}_buffer_{}_lr_{}_train_epoch_{}_batch_size_{}.pkl". \
        format(FLAGS.game_name, FLAGS.n_players, FLAGS.hidden_layer_size, FLAGS.data_number,
               FLAGS.learning_rate, epoch + 1, FLAGS.batch_size)

    return osp.join(result_dir, model_name)


# current_info_state, player_id, legal_actions, action, next_info_state, next_legal_actions,
# next_player, reward, done, chance_node
def add_transition(tran, action_number, replay_buffer):
    # state representation
    info_state = []
    next_info_state = []
    for i in range(len(tran[0])):
        info_state += tran[0][i]
        next_info_state += tran[4][i]
    # action
    one_hot_action = [1 if i == tran[3][0] else 0 for i in range(action_number)]

    # output: next state + next legal action + next player id + reward + done + chance node
    next_legal_action = [1 if i in tran[5] else 0 for i in range(action_number)]
    next_info_state += next_legal_action
    next_player_id = [1 if i == tran[6] else 0 for i in range(FLAGS.n_players)]
    next_info_state += next_player_id
    next_info_state += tran[7]
    next_info_state += tran[8]
    next_info_state += tran[9]

    replay_buffer["info_state"].append(info_state)
    replay_buffer["action"].append(one_hot_action)
    replay_buffer["next_info_state"].append(next_info_state)


def convert_data_to_replay_buffer(data, action_number, replay_buffer):
    for tran in data:
        add_transition(tran, action_number, replay_buffer)


def load_offline_data(offline_data_location, proportion):
    offline_data_location = offline_data_location + "/" + FLAGS.game_name + "_" + str(FLAGS.n_players) +  "_players_" + str(FLAGS.numdice) +"_numdice_exploit_{}/".format(FLAGS.exploitability)  + \
                            "seed_1_random_dataset_proportion_" + str(proportion) + "/data_number_" + str(FLAGS.data_number) + ".pkl"
    offline_data = torch.load(offline_data_location)
    return offline_data


def sample_mix_offline_data(proportion):
    return load_offline_data(FLAGS.mix_offline_data_location, proportion)


def data_to_tensor(replay_buffer):
    train_x = torch.tensor(replay_buffer["info_state"]).to(FLAGS.device)
    train_action = torch.tensor(replay_buffer["action"]).to(FLAGS.device)
    train_y = torch.tensor(replay_buffer["next_info_state"]).to(FLAGS.device)
    return train_x, train_action, train_y


def train(action_number, proportion):
    # load data
    offline_data = sample_mix_offline_data(proportion)
    replay_buffer = {"info_state": [], "action": [], "next_info_state": []}
    convert_data_to_replay_buffer(offline_data, action_number, replay_buffer)

    state_shape = len(replay_buffer["info_state"][0])
    output_size = state_shape + action_number + FLAGS.n_players * 2 + 2

    model = EnvModel(state_shape, action_number, FLAGS.hidden_layer_size, output_size).to(FLAGS.device)

    # train env model
    optimizer = torch.optim.SGD(model.parameters(), FLAGS.learning_rate)
    criterion = torch.nn.MSELoss().to(FLAGS.device)

    best_loss = 10000
    loss_list = []
    best_model = None

    train_x, train_action, train_y = data_to_tensor(replay_buffer)
    torch_dataset = Data.TensorDataset(train_x, train_action, train_y)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0)

    for epoch in range(FLAGS.train_epoch):
        epoch_total_loss = 0
        print("Epoch:", epoch)
        for step, (batch_x, batch_a, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            predict_y = model(batch_x, batch_a)
            batch_loss = criterion(predict_y, batch_y)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_total_loss += batch_loss.data
        loss_list.append(epoch_total_loss)

        if best_loss > epoch_total_loss:
            best_loss = epoch_total_loss
            best_model = model
        print(best_loss)

        # if (epoch + 1) % FLAGS.save_every == 0 and epoch != 0:
    torch.save(best_model, get_result_dir(FLAGS.train_epoch, proportion))


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
    game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players, "numdice": FLAGS.numdice, "dice_sides": 6})
    state = game.new_initial_state()
    chance_action = len(state.chance_outcomes())

    num_actions = max(game.num_distinct_actions(), chance_action)

    # train env model
    for proportion in range(11):
        train(num_actions, proportion)


if __name__ == "__main__":
    app.run(main)
