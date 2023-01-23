import numpy as np
from scipy.stats import poisson
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use('TkAgg')
# constants
LAMBDA_RENT_1 = 3
LAMBDA_RENT_2 = 4
LAMBDA_RETURN_1 = 3
LAMBDA_RETURN_2 = 2
THETA = 0.001
GAMMA = 0.9


# car states: represented as a list of tuples
car_states = []
for i in range(21):
    for j in range(21):
        car_states.append((i, j))

# car actions: action i = number of car transferred from lot 1 to lot 2 overnight
car_actions = [a for a in range(-5, 6)]

# calculate transition probabilities
transition_pr = {(s, a, sprime) : 0 for s in car_states for a in car_actions for sprime in car_states}

pr_rent_1 = np.zeros(21)
pr_rent_2 = np.zeros(21)
pr_return_1 = np.zeros(21)
pr_return_2 = np.zeros(21)

for i in range(1, 21):
    pr_rent_1[i] = poisson.pmf(i, LAMBDA_RENT_1)
    pr_rent_2[i] = poisson.pmf(i, LAMBDA_RENT_2)
    pr_return_1[i] = poisson.pmf(i, LAMBDA_RETURN_1)
    pr_return_2[i] = poisson.pmf(i, LAMBDA_RETURN_2)


def calculate_probability(s, a, sprime):

    s = (min(s[0] - a, 20), min(s[1] + a, 20))
    if s[0] < 0 or s[1] < 0:
        return 0

    ans_1 = 0.0
    for i in range(1, s[0]+1):
        ans_1 += pr_rent_1[i] * pr_return_1[min(20, sprime[0] - s[0] + i)]

    ans_2 = 0.0
    for i in range(1, s[1]+1):
        ans_2 += pr_rent_2[i] * pr_return_2[min(sprime[1] - s[1] + i, 20)]

    return ans_1 * ans_2



for s in car_states:
    for a in car_actions:
        for sprime in car_states:
            transition_pr[(s, a, sprime)] = calculate_probability(s, a, sprime)


# calculate rewards r(s, a, s')

def calculate_reward(s, a, sprime):

    s = (min(s[0] - a, 20), min(s[1] + a, 20))
    if s[0] < 0 or s[1] < 0:
        return 0

    if transition_pr[(s, a, sprime)] == 0:
        return 0

    ans_1 = 0.0
    for i in range(1, s[0]+1):
        ans_1 += pr_rent_1[i] * pr_return_1[min(20, sprime[0] - s[0] + i)] * 10 * i

    ans_2 = 0.0
    for i in range(1, s[1]+1):
        ans_2 += pr_rent_2[i] * pr_return_2[min(sprime[1] - s[1] + i, 20)] * 10 * i

    return ans_1 * ans_2 * transition_pr[(s, a, sprime)]

small = 1e9
big = 0
car_rewards = {(s, a, sprime) : 0 for s in car_states for a in car_actions for sprime in car_states}
for s in car_states:
    for a in car_actions:
        for sprime in car_states:
            car_rewards[(s, a, sprime)] = calculate_reward(s, a, sprime)
            big = max(big, car_rewards[(s, a, sprime)])
            small = min(small, car_rewards[(s, a, sprime)])

# car policy deterministic(initially doesn't move any cars)
policy = {s : 0 for s in car_states}

# value function
V = {s : 0 for s in car_states}

# policy evaluation

def policy_evaluation():
    global V
    delta = 1
    while delta > THETA:
        delta = 0
        v_temp = {s : 0 for s in car_states}
        for s in car_states:
            for sprime in car_states:
                v_temp[s] += transition_pr[(s, policy[s], sprime)] * (car_rewards[(s, policy[s], sprime)] + GAMMA * V[sprime])
            delta = max(delta, abs(v_temp[s] - V[s]))
        V = v_temp.copy()




# policy display
def display_policy(iter):
    ax = plt.figure().add_subplot(projection='3d')
    X = np.linspace(0, 20, 21)
    Y = np.linspace(0, 20, 21)
    Z = np.zeros(shape=(21,21))
    for i in range(21):
        for j in range(21):
            Z[i][j] = policy[(i, j)]
    ax.contour(X, Y, Z, extend3d=True, cmap=cm.coolwarm)
    ax.set_xlabel("Number of Cars in Lot 1")
    ax.set_ylabel("Number of Cars in Lot 2")
    ax.set_title("Policy {0}".format(iter))

    plt.show()



# policy iteration

def policy_iteration():
    stable = False
    i = 1
    while not stable:
        stable = True
        policy_evaluation()
        for s in car_states:
            old_a = policy[s]
            best_action = 0
            best_value = 0
            for a in car_actions:
                value = 0
                for sprime in car_states:
                    value += transition_pr[(s, a, sprime)] * (car_rewards[(s, a, sprime)] + GAMMA * V[sprime])
                if value > best_value:
                    best_value = value
                    best_action = a
            policy[s] = best_action
            if old_a != best_action:
                stable = False
        display_policy(i)
        i += 1


policy_iteration()
# print(policy)


