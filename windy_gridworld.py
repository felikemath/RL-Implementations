# In this example, we implement the SARSA algorithm on the Windy Gridworld problem

import random
from matplotlib import pyplot as plt

class WindyGridWorld:
    # class variables for UP, DOWN, LEFT, RIGHT
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    def __init__(self):
        self.states = []  # list of all states
        self.actions = dict()  # contains list of actions of each state (hashed by state)

        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.init_state = (3, 0)
        self.terminal_state = (3, 7)

        # Initialize states
        for i in range(7):
            for j in range(10):
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
        return 0 <= state[0] < 7 and 0 <= state[1] < 10


    def transition(self, state, action):
        '''
        :param state: state (i, j)
        :param action: action UP, DOWN, LEFT, RIGHT
        :return: state, reward
        '''
        new_state = None
        if action == WindyGridWorld.UP:
            new_state = (state[0] - 1, state[1])
        elif action == WindyGridWorld.DOWN:
            new_state = (state[0] + 1, state[1])
        elif action == WindyGridWorld.LEFT:
            new_state = (state[0], state[1] - 1)
        elif action == WindyGridWorld.RIGHT:
            new_state = (state[0], state[1] + 1)

        new_state = (max([new_state[0] - self.wind[state[1]], 0]), new_state[1])

        reward = 0 if state == (3, 7) else -1

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

def sarsa(env, episodes=100, eps=0.1, alpha=0.5, gamma=1):
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
        while s != env.terminal_state:
            s_prime, reward = env.transition(s, a)
            a_prime = epsilon_greedy(s_prime, q, env, eps)
            q[(s, a)] = q[(s, a)] + alpha * (reward + gamma * q[(s_prime, a_prime)] - q[(s, a)])
            s, a = s_prime, a_prime
            total_steps += 1
            steps += 1

            y.append(ep)

        print(f'Found terminal state in {steps} steps')
        print(f'Total steps: {total_steps}')

    x = list(range(total_steps))
    plt.plot(x, y)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.show()


if __name__ == '__main__':
    sarsa(WindyGridWorld(), episodes=180)
