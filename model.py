import numpy as np
from environment import InputState, ActionSpread, env, Q
from test import Test
import math


bLog = True
Reward = np.zeros((12, 9), dtype=float)


def InitRewards():
    inf = -float('inf')
    for sp in range(1, 5):  # When gaining
        for dp in range(-1, 2):
            for dask in range(-1, 2):
                for dbid in range(-1, 2):
                    s = InputState.index([sp, dp])
                    a = ActionSpread.index([dask, dbid])
                    Reward[s][a] = (sp + dask - dbid) * 0.25
    for sp in range(1, 5):  # When losing
        for dp in range(-1, 2):
            for dask in range(-1, 2):
                for dbid in range(-1, 2):
                    s = InputState.index([sp, dp])
                    a = ActionSpread.index([dask, dbid])
                    if (dp == 0) and ((dask == -1) or (dbid == 1)):
                        Reward[s][a] = -0.5
                    if (dp == -1) and (dbid != -1):
                        Reward[s][a] = -1.0
                    if (dp == 1) and (dask != 1):
                        Reward[s][a] = -1.0
    for sp in range(1, 5):  # When Spread overflow #Last
        for dp in range(-1, 2):
            for dask in range(-1, 2):
                for dbid in range(-1, 2):
                    s = InputState.index([sp, dp])
                    a = ActionSpread.index([dask, dbid])
                    if (sp + dask - dbid) > 4 or (sp + dask - dbid) < 1:
                        Reward[s][a] = inf
    if bLog:
        print("Rewards:", Reward)


def f(x):
    if x > 0:
        return (1 - env.k)*x
    else:
        return (1 + env.k)*x


points = [5, 50, 55, 65, 75, 85, 95, 100]
def QLearning(rewards, gamma=env.dis, num_episode=env.epochs):
    global Q
    Q = np.zeros(rewards.shape)
    all_states = np.arange(len(rewards))
    for i in range(num_episode):
        # initialize state
        initial_state = np.random.choice(all_states)
        action = np.random.choice(np.where(rewards[initial_state] != -float('inf'))[0])
        print(f(rewards[initial_state][action] + gamma * np.max(Q[action]) - Q[initial_state][action]))
        Q[initial_state][action] = Q[initial_state][action] + env.lRate * f(rewards[initial_state][action] + gamma * np.max(Q[action]) - Q[initial_state][action])
        if points.count(i+1) > 0 and bLog:
            Test(np.around(Q/np.max(Q)*100), i+1)
    Q = np.around(Q/np.max(Q)*100)
    if bLog:
        print("Final Q:", Q)
    return Q


def model(log=True):
    global bLog
    bLog = log
    InitRewards()
    return QLearning(Reward, 6)

