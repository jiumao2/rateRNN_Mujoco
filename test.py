import time

import gym
import numpy as np

env = gym.make('Swimmer-v4')
obs = env.reset()

T = 40
lag = 4
while True:
    for k in range(T):
        action = [0, 0]
        if k<T/2:
            action[0] = 1
        else:
            action[0] = -1

        if k>lag and k<T/2+lag:
            action[1] = 1
        else:
            action[1] = -1

        env.step(action)
        env.render()
        time.sleep(0.001)