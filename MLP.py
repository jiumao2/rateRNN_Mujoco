import time
# import torch
import matplotlib.pyplot as plt
import numpy as np
import gym
from get_hyperparams import get_hyperparams
from scipy.special import expit
from utils import make_random_weights
import torch.nn as nn
import torch

class net(nn.Module):
    def __init__(self,obs_dim,num_hidden_units,act_dim):
        super(net, self).__init__()
        self.obs_dim = obs_dim
        self.num_hidden_units = num_hidden_units
        self.action_dim = act_dim
        self.l1 = nn.Linear(obs_dim, num_hidden_units)
        self.a1 = nn.Tanh()
        self.l2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.a2 = nn.Tanh()
        self.l3 = nn.Linear(num_hidden_units, act_dim)
        self.a3 = nn.Tanh()

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)

        return x

    def set_weights(self, weights_mlp):
        p =0
        weights_in = torch.from_numpy(np.reshape(weights_mlp[:self.obs_dim * self.num_hidden_units],
                                     (self.num_hidden_units, self.obs_dim)))
        p+=self.obs_dim * self.num_hidden_units
        bias_in = torch.from_numpy(np.reshape(weights_mlp[
                               p:p + self.num_hidden_units],
                               (self.num_hidden_units,)))
        p+=self.num_hidden_units
        weights_hidden = torch.from_numpy(np.reshape(weights_mlp[:self.num_hidden_units * self.num_hidden_units],
                                     (self.num_hidden_units, self.num_hidden_units)))
        p+=self.num_hidden_units * self.num_hidden_units
        bias_hidden = torch.from_numpy(np.reshape(weights_mlp[
                               p:p + self.num_hidden_units],
                               (self.num_hidden_units,)))
        p+=self.num_hidden_units
        weights_out = torch.from_numpy(np.reshape(weights_mlp[p:p+self.action_dim * self.num_hidden_units],
                                      (self.action_dim, self.num_hidden_units)))
        p+=self.action_dim*self.num_hidden_units
        bias_out = torch.from_numpy(np.reshape(weights_mlp[p:],(self.action_dim,)))

        self.l1.weight = nn.Parameter(weights_in.float())
        self.l1.bias = nn.Parameter(bias_in.float())
        self.l2.weight = nn.Parameter(weights_hidden.float())
        self.l2.bias = nn.Parameter(bias_hidden.float())
        self.l3.weight = nn.Parameter(weights_out.float())
        self.l3.bias = nn.Parameter(bias_out.float())

    def get_flattened_weights(self):
        w0 = self.l1.weight.data.reshape((-1,))
        w1 = self.l1.bias.data.reshape((-1,))
        w2 = self.l2.weight.data.reshape((-1,))
        w3 = self.l2.bias.data.reshape((-1,))
        w4 = self.l3.weight.data.reshape((-1,))
        w5 = self.l3.bias.data.reshape((-1,))
        w = torch.cat((w0,w1,w2,w3,w4,w5),0).cpu().numpy()
        return w

    def initialize(self):
        nn.init.normal_(self.l1.weight, mean=0, std= 1/self.obs_dim)
        nn.init.constant_(self.l1.bias, 0)
        nn.init.normal_(self.l2.weight, mean=0, std= 1/self.num_hidden_units)
        nn.init.constant_(self.l2.bias, 0)
        nn.init.normal_(self.l3.weight, mean=0, std= 1/self.action_dim)
        nn.init.constant_(self.l3.bias, 0)

        return self.get_flattened_weights()

class MLP():
    def __init__(self, weights=None, env_name=None, obs_mean=None, obs_std=None, params=None, checkpoint=None):
        if checkpoint is None:
            assert env_name is not None
            self.weights = weights
            self.env = gym.make(env_name)

            if params is None:
                params = get_hyperparams()
            else:
                params = params

            if obs_mean is None:
                self.obs_mean = np.zeros(self.env.observation_space.shape[0])
            else:
                self.obs_mean = obs_mean

            if obs_std is None:
                self.obs_std = np.zeros(self.env.observation_space.shape[0])
            else:
                self.obs_std = obs_std

        else:
            self.env = gym.make(checkpoint["env_name"])
            params = checkpoint["params"]
            self.obs_mean = checkpoint["obs_mean"]
            self.obs_std = checkpoint["obs_std"]
            self.weights = checkpoint["weights"]


        self.params = params
        self.initiation_length = params["initiation_length"]
        self.warmup_steps = params['warmup_steps']
        self.min_reward_init = params['min_reward_init']

        self.obs = self.env.reset()
        self.reward = 0

        if type(self.env.action_space) is gym.spaces.box.Box:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = 1
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space

        self.num_hidden_units = params['num_hidden_units']
        self.done = False
        self.net = net(self.obs_dim, self.num_hidden_units, self.action_dim).cuda()
        if self.weights is not None:
            self.net.set_weights(self.weights)

    def reset(self):
        self.obs = self.env.reset()

    def step(self):
        output = (self.net.forward(self.obs).detach().numpy()+1)/2
        action = self.compute_action(output)
        self.obs, reward, done, _ = self.env.step(action)

        self.done = done
        self.reward += reward

    def compute_action(self, input):
        if type(self.action_space) is gym.spaces.box.Box:
            action = input * (self.action_space.high - self.action_space.low) + self.action_space.low
        else:
            action = min(int(np.floor(self.action_space.n * input) + self.action_space.start), self.action_space.n - 1)
        return action

    def compute_observation(self):
        return (self.obs - self.obs_mean) / self.obs_std

    def initiation_mlp(self):
        obs_all = np.zeros((self.initiation_length, self.obs_dim))
        for k in range(self.initiation_length):
            obs, reward, done, info = self.env.step(self.action_space.sample())
            if done:
                self.obs = self.env.reset()
            obs_all[k, :] = obs

        self.obs_mean = obs_all.mean(axis=0)
        self.obs_std = obs_all.std(axis=0)

        self.obs = self.env.reset()
        weights = self.net.initialize()

        print("Obversation mean: ", self.obs_mean)
        print("Obversation std: ", self.obs_std)
        return (self.obs_mean, self.obs_std, weights)


    def visualize(self):
        self.reset()
        print('MLP policy!')
        while True:
            obs = self.compute_observation()
            hidden = self.activate(np.matmul(self.weights_in, obs) + self.bias)
            out = self.activate(np.matmul(self.weights_out, hidden))
            action = self.compute_action(out)
            self.obs, reward, done, _ = self.env.step(action)
            self.env.render()
            time.sleep(0.001)

    def visualize_weight(self):
        plt.imshow(self.weights)
        plt.title('weights')
        plt.colorbar()
        plt.show()


    def get_total_reward(self):
        return self.reward





