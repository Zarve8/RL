import numpy as np
import gym


ActionSpread = [[-1, -1], [-1, 0], [-1, 1],  # dAsk, dBid
                [0, -1], [0, 0], [0, 1],
                [1, -1], [1, 0], [1, 1]]

InputState = [[1, -1], [2, -1], [3, -1], [4, -1],  # SP, dP
              [1, 0], [2, 0], [3, 0], [4, 0],
              [1, 1], [2, 1], [3, 1], [4, 1]]

Q = np.zeros((12, 9), dtype=float)
GymEnv = gym.make('FrozenLake-v1')

class Env:
    aMove = 0.105
    aInf = 0.526
    aUn = 0.132
    dis = 0.9
    period = 250
    start = 100
    startAsk = 101
    startBid = 99
    lRate = 0.04
    epochs = 4000
    samples = 2000
    greed = 0.9
    wInv = 0.1
    wPro = 0.8
    wSp = 0.1
    k = 0.8

env = Env()
