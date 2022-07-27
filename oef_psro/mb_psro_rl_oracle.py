import numpy as np
from open_spiel.python.algorithms.psro_v2 import optimization_oracle
from open_spiel.python.algorithms.psro_v2 import utils


def update_episodes_per_oracles(episodes_per_oracle, played_policies_indexes):
    for player_index, policy_index in played_policies_indexes:
        episodes_per_oracle[player_index][policy_index] += 1
    return episodes_per_oracle


def freeze_all(policies_per_player):
    for policies in policies_per_player:
        for pol in policies:
            pol.freeze()


def random_count_weighted_choice(count_weight):
    indexes = list(range(len(count_weight)))
    p = np.array([1 / (weight + 1) for weight in count_weight])
    p /= np.sum(p)
    chosen_index = np.random.choice(indexes, p=p)
    return chosen_index


class RLOracle(optimization_oracle.AbstractOracle):
    def __init__(self,
                 env,
                 env_model,
                 game_length,
                 best_response_class,
                 best_response_kwargs,
                 number_training_episodes=1e3,
                 self_play_proportion=0.0,
                 **kwargs):
        self._env = env
        self.env_model = env_model
        self.game_length = game_length

        self._best_response_class = best_response_class
        self._best_response_kwargs = best_response_kwargs

        self._self_play_proportion = self_play_proportion
        self._number_training_episodes = number_training_episodes

        super(RLOracle, self).__init__(**kwargs)

    def sample_episode(self, unused_time_step, agents, is_evaluation=False):
        time_step = self._env.reset()
        cumulative_rewards = 0.0
        step = 0
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step, is_evaluation=is_evaluation)
            action_list = [agent_output.action]
            time_step = self.env_model.step(time_step, action_list)
            cumulative_rewards += np.array(time_step.rewards)
            step += 1

        if not is_evaluation:
            for agent in agents:
                agent.step(time_step)

        return cumulative_rewards

    def _has_terminated(self, episodes_per_oracle):
        return np.all(episodes_per_oracle.reshape(-1) > self._number_training_episodes)

    def sample_policies_for_episode(self, new_policies, training_parameters, episodes_per_oracle, strategy_sampler):
        num_players = len(training_parameters)

        # Prioritizing players that haven't had as much training as the others.
        episodes_per_player = [sum(episodes) for episodes in episodes_per_oracle]
        chosen_player = random_count_weighted_choice(episodes_per_player)

        # Uniformly choose among the sampled player.
        agent_chosen_ind = np.random.randint(0, len(training_parameters[chosen_player]))
        agent_chosen_dict = training_parameters[chosen_player][agent_chosen_ind]
        new_policy = new_policies[chosen_player][agent_chosen_ind]

        # Sample other players' policies.
        total_policies = agent_chosen_dict["total_policies"]
        probabilities_of_playing_policies = agent_chosen_dict["probabilities_of_playing_policies"]
        episode_policies = strategy_sampler(total_policies, probabilities_of_playing_policies)

        live_agents_player_index = [(chosen_player, agent_chosen_ind)]

        for player in range(num_players):
            if player == chosen_player:
                episode_policies[player] = new_policy
                assert not new_policy.is_frozen()
            else:
                if np.random.binomial(1, self._self_play_proportion):
                    agent_index = random_count_weighted_choice(episodes_per_oracle[player])
                    self_play_agent = new_policies[player][agent_index]
                    episode_policies[player] = self_play_agent
                    live_agents_player_index.append((player, agent_index))
                else:
                    assert episode_policies[player].is_frozen()

        return episode_policies, live_agents_player_index

    def _rollout(self, agents):
        self.sample_episode(None, agents, is_evaluation=False)

    def generate_new_policies(self, training_parameters):
        new_policies = []
        for player in range(len(training_parameters)):
            player_parameters = training_parameters[player]
            new_pols = []
            for param in player_parameters:
                current_pol = param["policy"]
                if isinstance(current_pol, self._best_response_class):
                    new_pol = current_pol.copy_with_noise(self._kwargs.get("sigma", 0.0))
                else:
                    new_pol = self._best_response_class(self._env, player, **self._best_response_kwargs)
                    new_pol.unfreeze()
                new_pols.append(new_pol)
            new_policies.append(new_pols)
        return new_policies

    def __call__(self, game, training_parameters, strategy_sampler=utils.sample_strategy,
                 **oracle_specific_execution_kwargs):

        episodes_per_oracle = [[0 for _ in range(len(player_params))] for player_params in training_parameters]
        episodes_per_oracle = np.array(episodes_per_oracle)

        new_policies = self.generate_new_policies(training_parameters)

        while not self._has_terminated(episodes_per_oracle):
            agents, indexes = self.sample_policies_for_episode(new_policies, training_parameters, episodes_per_oracle,
                                                               strategy_sampler)
            self._rollout(agents)
            episodes_per_oracle = update_episodes_per_oracles(episodes_per_oracle, indexes)

        freeze_all(new_policies)
        return new_policies
