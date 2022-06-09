import itertools
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms.psro_v2 import abstract_meta_trainer
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.algorithms.psro_v2 import utils

TRAIN_TARGET_SELECTORS = {"": None,
                          "rectified": strategy_selectors.rectified_selector}


class PSROSolver(abstract_meta_trainer.AbstractMetaTrainer):
    def __init__(self,
                 game,
                 oracle,
                 sims_per_entry,
                 initial_policies=None,
                 rectifier="",
                 training_strategy_selector=None,
                 meta_strategy_method="alpharank",
                 sample_from_marginals=False,
                 number_policies_selected=1,
                 n_noisy_copies=0,
                 alpha_noise=0.0,
                 beta_noise=0.0,
                 **kwargs):

        self.env = oracle._env
        self.env_model = oracle.env_model

        self._sims_per_entry = sims_per_entry
        print("Using {} sims per entry.".format(sims_per_entry))

        self._rectifier = TRAIN_TARGET_SELECTORS.get(rectifier, None)
        self._rectify_training = self._rectifier
        print("Rectifier : {}".format(rectifier))

        self._meta_strategy_probabilities = np.array([])
        self._non_marginalized_probabilities = np.array([])

        print("Perturbating oracle outputs : {}".format(n_noisy_copies > 0))
        self._n_noisy_copies = n_noisy_copies
        self._alpha_noise = alpha_noise
        self._beta_noise = beta_noise

        self._policies = []
        self._new_policies = []

        if not meta_strategy_method or meta_strategy_method == "alpharank":
            meta_strategy_method = utils.alpharank_strategy

        print("Sampling from marginals : {}".format(sample_from_marginals))
        self.sample_from_marginals = sample_from_marginals

        super(PSROSolver, self).__init__(
            game,
            oracle,
            initial_policies,
            meta_strategy_method,
            training_strategy_selector,
            number_policies_selected=number_policies_selected,
            **kwargs)

    def _initialize_policy(self, initial_policies):
        self._policies = [[] for k in range(self._num_players)]
        self._new_policies = [([initial_policies[k]] if initial_policies else [policy.UniformRandomPolicy(self._game)])
                              for k in range(self._num_players)]

    def _initialize_game_state(self):
        effective_payoff_size = self._game_num_players
        self._meta_games = [np.array(utils.empty_list_generator(effective_payoff_size))
                            for _ in range(effective_payoff_size)]
        self.update_empirical_gamestate(seed=None)

    def get_joint_policy_ids(self):
        return utils.get_strategy_profile_ids(self._meta_games)

    def get_joint_policies_from_id_list(self, selected_policy_ids):
        policies = self.get_policies()
        selected_joint_policies = utils.get_joint_policies_from_id_list(self._meta_games, policies, selected_policy_ids)
        return selected_joint_policies

    def update_meta_strategies(self):
        if self.symmetric_game:
            self._policies = self._policies * self._game_num_players

        self._meta_strategy_probabilities, self._non_marginalized_probabilities = \
            self._meta_strategy_method(solver=self, return_joint=True)

        if self.symmetric_game:
            self._policies = [self._policies[0]]
            self._meta_strategy_probabilities = [self._meta_strategy_probabilities[0]]

    def get_policies_and_strategies(self):
        sample_strategy = utils.sample_strategy_marginal
        probabilities_of_playing_policies = self.get_meta_strategies()

        if self._rectify_training or not self.sample_from_marginals:
            sample_strategy = utils.sample_strategy_joint
            probabilities_of_playing_policies = self._non_marginalized_probabilities

        total_policies = self.get_policies()
        return sample_strategy, total_policies, probabilities_of_playing_policies

    def _restrict_target_training(self,
                                  current_player,
                                  ind,
                                  total_policies,
                                  probabilities_of_playing_policies,
                                  restrict_target_training_bool,
                                  epsilon=1e-12):

        true_shape = tuple([len(a) for a in total_policies])
        if not restrict_target_training_bool:
            return probabilities_of_playing_policies
        else:
            kept_probas = self._rectifier(self, current_player, ind)
            probability = probabilities_of_playing_policies.reshape(true_shape)
            probability = probability * kept_probas
            prob_sum = np.sum(probability)

            if prob_sum <= epsilon:
                probability = probabilities_of_playing_policies
            else:
                probability /= prob_sum

            return probability

    def update_agents(self):
        used_policies, used_indexes = self._training_strategy_selector(self, self._number_policies_selected)
        (sample_strategy, total_policies, probabilities_of_playing_policies) = self.get_policies_and_strategies()
        training_parameters = [[] for _ in range(self._num_players)]

        for current_player in range(self._num_players):
            if self.sample_from_marginals:
                currently_used_policies = used_policies[current_player]
                current_indexes = used_indexes[current_player]
            else:
                currently_used_policies = [joint_policy[current_player] for joint_policy in used_policies]
                current_indexes = used_indexes[current_player]

            for i in range(len(currently_used_policies)):
                pol = currently_used_policies[i]
                ind = current_indexes[i]
                new_probabilities = self._restrict_target_training(current_player, ind, total_policies,
                                                                   probabilities_of_playing_policies,
                                                                   self._rectify_training)

                new_parameter = {"policy": pol, "total_policies": total_policies,
                                 "current_player": current_player,
                                 "probabilities_of_playing_policies": new_probabilities}

                training_parameters[current_player].append(new_parameter)

        if self.symmetric_game:
            self._policies = self._game_num_players * self._policies
            self._num_players = self._game_num_players
            training_parameters = [training_parameters[0]]

        self._new_policies = self._oracle(self._game,
                                          training_parameters,
                                          strategy_sampler=sample_strategy,
                                          using_joint_strategies=self._rectify_training or not self.sample_from_marginals)

        if self.symmetric_game:
            self._policies = [self._policies[0]]
            self._num_players = 1

    def update_empirical_gamestate(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        assert self._oracle is not None

        updated_policies = [self._policies[k] + self._new_policies[k] for k in range(self._num_players)]

        total_number_policies = [len(updated_policies[k]) for k in range(self._num_players)]
        number_older_policies = [len(self._policies[k]) for k in range(self._num_players)]
        number_new_policies = [len(self._new_policies[k]) for k in range(self._num_players)]

        # Initializing the matrix with nans to recognize unestimated states.
        meta_games = [np.full(tuple(total_number_policies), np.nan) for k in range(self._num_players)]

        # Filling the matrix with already-known values.
        older_policies_slice = tuple([slice(len(self._policies[k])) for k in range(self._num_players)])
        for k in range(self._num_players):
            meta_games[k][older_policies_slice] = self._meta_games[k]

        # Filling the matrix for newly added policies.
        for current_player in range(self._num_players):
            # Only iterate over new policies for current player ; compute on every
            # policy for the other players.
            range_iterators = [range(total_number_policies[k]) for k in range(current_player)] + \
                              [range(number_new_policies[current_player])] + \
                              [range(total_number_policies[k]) for k in range(current_player + 1, self._num_players)]

            for current_index in itertools.product(*range_iterators):
                used_index = list(current_index)
                used_index[current_player] += number_older_policies[current_player]

                if np.isnan(meta_games[current_player][tuple(used_index)]):

                    estimated_policies = [updated_policies[k][current_index[k]] for k in range(current_player)] + \
                                         [self._new_policies[current_player][current_index[current_player]]] + \
                                         [updated_policies[k][current_index[k]] for k in range(current_player + 1, self._num_players)]

                    utility_estimates = self.sample_episodes(estimated_policies, self._sims_per_entry)
                    for k in range(self._num_players):
                        meta_games[k][tuple(used_index)] = utility_estimates[k]

        self._meta_games = meta_games
        self._policies = updated_policies
        return meta_games

    def sample_episodes(self, policies, num_episodes):
        totals = np.zeros(self._num_players)
        for _ in range(num_episodes):
            totals += sample_episode(self.env.reset(), self.env_model, policies).reshape(-1)
        return totals / num_episodes

    def get_meta_game(self):
        return self._meta_games

    @property
    def meta_games(self):
        return self._meta_games

    def get_policies(self):
        policies = self._policies
        if self.symmetric_game:
            policies = self._game_num_players * self._policies
        return policies

    def get_and_update_non_marginalized_meta_strategies(self, update=True):
        if update:
            self.update_meta_strategies()
        return self._non_marginalized_probabilities

    def get_strategy_computation_and_selection_kwargs(self):
        return self._strategy_computation_and_selection_kwargs


def sample_episode(time_step, env_model, policies):
    if time_step.last():
        return np.array(time_step.rewards, dtype=np.float32)

    player = time_step.observations["current_player"]
    agent_output = policies[player].step(time_step, is_evaluation=True)
    action_list = [agent_output.action]
    time_step = env_model.step(time_step, action_list)

    return sample_episode(time_step, env_model, policies)
