import time
import gym
import utils
from get_hyperparams import get_hyperparams
from rateRNN import nonNN as Model
import numpy as np
from matplotlib import pyplot as plt
import utils
import pickle
import cv2

log_dir = r"./results/Swimmer-v4_20221012_214002_nonNN_mask_3_4_5_6_7/"

# utils.visualize_log(log_dir+"log.pickle")

data = utils.load_pickle(log_dir + "Swimmer-v4_checkpoint_iter161.pickle")
temp = data['weights']
# temp[-5:] = 0
# temp[-5-8:-7] = 0
# temp[-16] = 0
temp[-5:] = 0
temp[-5-8:-8] = 0
data['weights'] = temp
print('weights of node 0: ', temp[-16], temp[-8])
print('weights of node 1: ', temp[-15], temp[-7])
print('weights of node 2: ', temp[-14], temp[-6])

env = gym.make('Swimmer-v4')
obs_rec = []
obs = env.reset()
obs_rec.append(obs)
reward_total = 0
for _ in range(1000):
    action = [0, 0]
    if obs[0]*temp[-16] + obs[1]*temp[-15] + obs[2]*temp[-14] > 0:
        action[0] = 1
    else:
        action[0] = -1

    if obs[0]*temp[-8] + obs[1]*temp[-7] + obs[2]*temp[-6] > 0:
        action[1] = 1
    else:
        action[1] = -1

    # action = [obs[1]*temp[-15] + obs[2]*temp[-14], obs[1]*temp[-7] + obs[2]*temp[-6]]

    obs, reward, _, _ = env.step(action)
    obs_rec.append(obs)
    reward_total += reward

    # env.render()
    # time.sleep(0.001)

print('Reward', reward_total)

fig = plt.figure(figsize=[10, 10])
ax = plt.axes(projection='3d')

x = np.linspace(-2,2,10)
y = np.linspace(-2,2,10)
X, Y = np.meshgrid(x, y)
Z = -(X*temp[-16] + Y*temp[-15])/temp[-14]
ax.plot_wireframe(X, Y, Z, color='green')

x = np.linspace(-2,2,10)
y = np.linspace(-2,2,10)
X, Y = np.meshgrid(x, y)
Z = -(X*temp[-8] + Y*temp[-7])/temp[-6]
ax.plot_wireframe(X, Y, Z, color='red')

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

ax.plot3D(np.array(obs_rec)[:, 0], np.array(obs_rec)[:, 1], np.array(obs_rec)[:, 2])
ax.set_xlabel('Observation 0')
ax.set_ylabel('Observation 1')
ax.set_zlabel('Observation 2')

plt.show()
# plt.savefig('fig.jpg')


