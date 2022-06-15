# Offline Equilibrium Finding - OEF
Repository for the submission of NeurIPS Datasets and Benchmarks Track 2022.

# Dataset

The dataset in this repository includes three types of datasets for every game: random dataset, expert dataset and learning dataset. All dataset are avaiable at . The data entry in dataset is [current_game_state, player_id, legal_actions, action, next_game_state, next_legal_actions, next_player, reward, done, chance_node]. 

current_game_state: a list of every player's infomation state list

player_id: the player should take actions at curretn game state

legal_actions: the available action set of current game state

action: the selected action

next_game_state: a list of every player's infomation state list after excuting the action

next_legal_actions: the available action set of next game state

next_player: the player should take actions at next game state

reward: the rewards of every player of next game state

done: whether the next game state is a end state

chance_node: whether the next game state is a chance state

# How to run the code

Behavior Cloning Algorithm: run the train_bc_policy.py file to get the behavior cloning policy by modifying the game and corresponding dataset in that file

Model-based Algorithm: first run the train_env_model.py file to train the environment model by modifying the game and corresponding dataset in that file and then run OEF-CFR or OEF-PSRO algorithm to get the model-based policy based on the trained environment model

OEF-CFR: run the 
