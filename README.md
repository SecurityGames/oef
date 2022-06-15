# Offline Equilibrium Finding - OEF
Repository for the submission of NeurIPS Datasets and Benchmarks Track 2022.

# Dataset

The dataset in this repository includes three types of datasets for every game: random dataset, expert dataset and learning dataset. All dataset are avaiable at . The data entry in dataset is [current_game_state, player_id, legal_actions, action, next_game_state, next_legal_actions, next_player, reward, done, chance_node]. 
current_game_state: a list of every player's infomation state list
player_id: the player id 
legal_actions: the available action set of current game state
action: the selected action
next_game_state: a list of every player's infomation state list after excuting the action
next_legal_actions: the available action set of next game state
next_player: the player id of 
reward: the rewards of every player of next game state
done: whether the next game state is a end state
chance_node: whether the next game state is a chance state
