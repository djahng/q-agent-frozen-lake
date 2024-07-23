import numpy as np


class Agent:
    def __init__(self, n_states, n_actions, learn_rate, gamma,
                 epsilon, epsilon_min, epsilon_decrement):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = epsilon_decrement

        self.Q = {}
        self.initialize_Q()

    def initialize_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            # Explore
            return np.random.choice([i for i in range(self.n_actions)])
        else:
            # Greedy
            Qs = [self.Q[(state, a)] for a in range(self.n_actions)]
            action_max = np.argmax(Qs)

            return action_max

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decrement
        else:
            self.epsilon = self.epsilon_min

    def learn(self, state, action, reward, state_):
        actions = [self.Q[(state_, a)] for a in range(self.n_actions)]
        a_max = np.argmax(actions)

        self.Q[(state, action)] += self.learn_rate * \
                                  (reward + self.gamma *
                                   self.Q[(state_, a_max)] -
                                   self.Q[(state, action)])

        self.decrement_epsilon()
