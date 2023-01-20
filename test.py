from environment import ActionSpread, InputState, env
import numpy as np
import random as rd
from matplotlib import pyplot as plt


Q = 0


def ChooseAction(state):
    line = Q[state]
    action = 0
    res = line[0]
    for i in range(0, len(line)):
        if line[i] > res:
            res = line[i]
            action = i
    return action


def MakeAction(sp, dp, ask, bid):
    stateIndex = InputState.index([sp, dp])
    action = ActionSpread[ChooseAction(stateIndex)]
    #print(sp, dp, ask, bid, stateIndex, ChooseAction(stateIndex), action)
    ask += action[0]
    bid += action[1]
    return ask, bid


def CalcProfit(p, ask, bid, inv):
    profit = env.aUn*(ask - bid) / 2.0
    if rd.random() > 0.5:
        inv += env.aUn
    else:
        inv -= env.aUn
    if ask < p:
        profit -= env.aInf*(p - ask)
        inv += env.aInf
    if bid > p:
        inv -= env.aInf
        profit -= env.aInf*(bid - p)
    return profit, inv


def GenerateWiener():
    w = [0] * env.period
    for i in range(0, env.period):
        v = rd.random()
        if v < env.aMove:
            w[i] = -1
        elif v > 1 - env.aMove:
            w[i] = 1
    return w


def GeneratePrice(w):
    p = [0] * env.period
    s = env.start
    for i in range(0, env.period):
        s += w[i]
        p[i] = s
    return p


def Test(_Q, epoch):
    global Q
    Q = _Q
    w = GenerateWiener()
    p = GeneratePrice(w)
    ask = env.startAsk
    bid = env.startBid
    sumR = 0.0
    sp = 2
    inv = 0.0
    Spreads = [0] * env.period
    Profit = [0] * env.period
    Inventory = [0] * env.period
    R = [0] * env.period
    for i in range(0, env.period):
        ask, bid = MakeAction(sp, w[i], ask, bid)
        rew, inv = CalcProfit(p[i], ask, bid, inv)
        sumR = rew + sumR*env.dis
        sp = ask - bid
        Spreads[i] = sp
        Profit[i] = sumR
        Inventory[i] = inv
        R[i] = sumR * env.wPro - abs(inv) * env.wInv - sp * env.wSp
    if True:
        plt.clf()
        plt.title(f'Trajectory [Episode{epoch}]')
        plt.plot([i for i in range(0, env.period)], p)
        plt.show()
    if True:
        plt.clf()
        plt.title(f'Profit [Episode{epoch}]')
        plt.plot([i for i in range(0, env.period)], R)
        plt.show()
    if True:
        plt.clf()
        plt.title(f'Spread [Episode{epoch}]')
        plt.plot([i for i in range(0, env.period)], Spreads)
        plt.show()
    if True:
        plt.clf()
        plt.title(f'Inventory [Episode{epoch}]')
        plt.plot([i for i in range(0, env.period)], Inventory)
        plt.show()
    return sumR


