# In this example, we compare the SARSA algorithm and Q-Learning on the Cliff Walking problem

import random
from matplotlib import pyplot as plt
from windy_gridworld import WindyGridWorld

class CliffWorld:
    # class variables for UP, DOWN, LEFT, RIGHT
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    def __init__(self):
        self.states = []  # list of all states
        self.actions = dict()  # contains list of actions of each state (hashed by state)

        self.init_state = (3, 0)
        self.terminal_state = (3, 11)

        # Initialize states
        for i in range(4):
            for j in range(12):
                self.states.append((i, j))

                self.actions[(i, j)] = []

                for action in range(4):
                    if self.valid_state(self.transition((i, j), action)[0]):
                        self.actions[(i, j)].append(action)


    def valid_state(self, state):
        '''
        :param state: state (i, j)
        :return: bool
        '''
        return 0 <= state[0] < 4 and 0 <= state[1] < 12


    def transition(self, state, action):
        '''
        :param state: state (i, j)
        :param action: action UP, DOWN, LEFT, RIGHT
        :return: state, reward
        '''
        new_state = None
        if action == CliffWorld.UP:
            new_state = (state[0] - 1, state[1])
        elif action == CliffWorld.DOWN:
            new_state = (state[0] + 1, state[1])
        elif action == CliffWorld.LEFT:
            new_state = (state[0], state[1] - 1)
        elif action == CliffWorld.RIGHT:
            new_state = (state[0], state[1] + 1)

        reward = -1
        if new_state[0] == 3 and 0 < new_state[1] < 11:
            reward = -100
            new_state = self.init_state

        return new_state, reward

def epsilon_greedy(s, q, env, eps):
    # choose action with respect to epsilon greedy algorithm
    rand = random.random()
    a = None
    action_dict = dict()
    for act in env.actions[s]:
        action_dict[act] = q[(s, act)]

    if rand > eps:
        # greedy option
        a = max(action_dict, key=action_dict.get)
    else:
        # exploration option
        act_list = []
        for act in env.actions[s]:
            if q[(s, act)] < max(action_dict.values()):
                act_list.append(act)

        if len(act_list) == 0:
            act_list = list(action_dict.keys())

        a = random.choice(act_list)

    return a

def sarsa(env, episodes=100, eps=0.1, alpha=0.5, gamma=1, plot=False):
    # Initialize Q(S, A) arbitrarily and Q(terminal state, dot) = 0
    q = dict()  # state action value function

    for state in env.states:
        for action in env.actions[state]:
            q[(state, action)] = 0

    total_steps = 0
    y = []
    for ep in range(episodes):
        s = env.init_state
        a = epsilon_greedy(s, q, env, eps)
        steps = 0
        ep_reward = 0
        while s != env.terminal_state:
            s_prime, reward = env.transition(s, a)
            ep_reward += reward
            a_prime = epsilon_greedy(s_prime, q, env, eps)
            q[(s, a)] = q[(s, a)] + alpha * (reward + gamma * q[(s_prime, a_prime)] - q[(s, a)])
            s, a = s_prime, a_prime
            total_steps += 1
            steps += 1

        print(f'Found terminal state in {steps} steps')
        print(f'Total steps: {total_steps}')
        print(f'Reward: {ep_reward}')

        y.append(ep_reward)

    if plot:
        x = list(range(episodes))
        plt.plot(x, y)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.ylim([-500, 0])
        plt.show()

    return q


def qlearning(env, episodes=100, eps=0.1, alpha=0.5, gamma=1, plot=False):
    # Initialize Q(S, A) arbitrarily and Q(terminal state, dot) = 0
    q = dict()  # state action value function

    for state in env.states:
        for action in env.actions[state]:
            q[(state, action)] = 0

    total_steps = 0
    y = []
    for ep in range(episodes):
        s = env.init_state

        steps = 0
        ep_reward = 0
        while s != env.terminal_state:
            a = epsilon_greedy(s, q, env, eps)
            s_prime, reward = env.transition(s, a)
            ep_reward += reward
            best_action_value = max([q[(s_prime, a_prime)] for a_prime in env.actions[s_prime]])
            q[(s, a)] = q[(s, a)] + alpha * (reward + gamma * best_action_value - q[(s, a)])
            s = s_prime
            total_steps += 1
            steps += 1

        print(f'Found terminal state in {steps} steps')
        print(f'Total steps: {total_steps}')
        print(f'Reward: {ep_reward}')

        y.append(ep_reward)

    if plot:
        x = list(range(episodes))
        plt.plot(x, y)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

    return q



def sample(env, q):
    steps = 0
    reward = 0

    s = env.init_state

    while s != env.terminal_state:
        a = epsilon_greedy(s, q, env, eps=0)
        s_prime, r = env.transition(s, a)
        reward += r
        steps += 1
        s = s_prime
        print(s_prime)

    print(f'Greedy steps: {steps}')
    print(f'Greedy reward: {reward}')


if __name__ == '__main__':
    q = qlearning(CliffWorld(), episodes=500)
    sample(CliffWorld(), q)

