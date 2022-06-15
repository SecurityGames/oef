import os
import torch
import random
import pyspiel
import numpy as np
from absl import app
import os.path as osp
from absl import flags
import torch.utils.data as Data
from network.policy_network import BehaviorPolicyModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 1, "Seed.")

flags.DEFINE_string("game_name", "liars_dice", "Game name.")
flags.DEFINE_integer("n_players", 2, "The number of players.")
flags.DEFINE_integer("numdice", 1, "The number of players.")

# offline data location
flags.DEFINE_string("mix_offline_data_location",
                    "dataset/mix_offline_dataset",
                    "offline data location")

# record results location
flags.DEFINE_string("result_data_location", "mix_offline_dataset_behavior_clone_policy", "dataset class")
flags.DEFINE_float("exploitability", 0.03779695063020872, "the location of trained expert policy")

# train env model
flags.DEFINE_float("learning_rate", 0.05, "train env model learning rate.")
flags.DEFINE_integer("hidden_layer_size", 64, "env model Hidden layer size")
flags.DEFINE_integer("data_number", 50000, "The number of data.")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("train_epoch", 5000, "batch size")
# flags.DEFINE_integer("save_every", int(1e3), "save model every")

flags.DEFINE_string("device", "cpu", "device type")
FLAGS.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_result_dir(epoch, player, proportion):
    result_dir = str(FLAGS.result_data_location) + "/" + FLAGS.game_name + '_' + str(FLAGS.n_players) + "_players_" + str(FLAGS.numdice) + "_numdice"
    sub_folder = str(player) + "_player_policy_proportion_" + str(proportion)

    result_dir = osp.join(result_dir, sub_folder)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    model_name = "seed_{}_game_{}_players_{}_hidden_layer_{}_buffer_{}_lr_{}_train_epoch_{}_batch_size_{}_policy.pkl". \
        format(FLAGS.seed, FLAGS.game_name, FLAGS.n_players, FLAGS.hidden_layer_size, FLAGS.data_number,
               FLAGS.learning_rate, epoch, FLAGS.batch_size)

    return osp.join(result_dir, model_name)


def add_transition(prev_info_state, action, replay_buffer, player):
    # state representation
    info_state = prev_info_state[player]
    one_hot_action = action[0]
    replay_buffer["info_state"].append(info_state)
    replay_buffer["action"].append(one_hot_action)


def convert_data_to_replay_buffer(data, replay_buffer, player):
    for tran in data:
        add_transition(tran[0], tran[3], replay_buffer, player)


def load_offline_data(offline_data_location, player, proportion):
    offline_data_location = offline_data_location + "/" + FLAGS.game_name + "_" + str(FLAGS.n_players) + "_players_" + str(FLAGS.numdice) +"_numdice_exploit_{}/".format(FLAGS.exploitability) +\
                            "seed_1_random_dataset_proportion_" + str(proportion) + "/data_number_" + str(FLAGS.data_number) + ".pkl"
    offline_data = torch.load(offline_data_location)

    # get one player's data
    offline = []
    for data in offline_data:
        if data[1] == player:
            offline.append(data)

    return offline


def sample_mix_offline_data(player, proportion):
    mix_offline = load_offline_data(FLAGS.mix_offline_data_location, player, proportion)
    return mix_offline


def data_to_tensor(replay_buffer):
    train_x = torch.tensor(replay_buffer["info_state"]).to(FLAGS.device)
    train_action = torch.tensor(replay_buffer["action"]).to(FLAGS.device)
    return train_x, train_action


def train(action_number, player, proportion):
    # load data
    offline_data = sample_mix_offline_data(player, proportion)
    replay_buffer = {"info_state": [], "action": []}
    convert_data_to_replay_buffer(offline_data, replay_buffer, player)

    state_shape = len(replay_buffer["info_state"][0])

    model = BehaviorPolicyModel(state_shape, FLAGS.hidden_layer_size, action_number).to(FLAGS.device)

    # train policy model
    optimizer = torch.optim.SGD(model.parameters(), FLAGS.learning_rate)
    criterion = torch.nn.CrossEntropyLoss().to(FLAGS.device)

    best_loss = 10000
    best_model = None

    train_x, train_action = data_to_tensor(replay_buffer)
    torch_dataset = Data.TensorDataset(train_x, train_action)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0)
    epoch_total_loss_list = []
    for epoch in range(FLAGS.train_epoch):
        epoch_total_loss = 0
        print("Epoch:", epoch)
        for step, (batch_x, batch_a) in enumerate(loader):
            optimizer.zero_grad()
            predict_y = model(batch_x)
            batch_loss = criterion(predict_y, batch_a)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_total_loss += batch_loss.data.item()
        epoch_total_loss_list.append(epoch_total_loss)
        print(epoch_total_loss)

        if best_loss > epoch_total_loss:
            best_loss = epoch_total_loss
            best_model = model

        # save model
        # if (epoch + 1) % FLAGS.save_every == 0 and epoch != 0:
    torch.save(best_model, get_result_dir(FLAGS.train_epoch, player, proportion))


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
    # load liar's dice game
    game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players, "numdice": FLAGS.numdice, "dice_sides": 6})
    # load poker game
    # game = pyspiel.load_game(FLAGS.game_name, {"players": FLAGS.n_players})
    # load phantom ttt game
    # game = pyspiel.load_game(FLAGS.game_name, {"obstype": "reveal-nothing"})

    num_actions = game.num_distinct_actions()
    train(num_actions, 1, 3)

    # train policy model
    for proportion in range(11):
        for player in range(FLAGS.n_players):
            train(num_actions, player, proportion)
            print("Finish train player {} proportion {}".format(player, proportion))


if __name__ == "__main__":
    app.run(main)
