import time
# import torch
import matplotlib.pyplot as plt
import numpy as np
import gym
from get_hyperparams import get_hyperparams
from scipy.special import expit
from utils import make_random_weights
from gym.wrappers.monitoring.video_recorder import VideoRecorder
# from torch.nn import functional as F


class rateRNN():
    def __init__(self, weights=None, env_name=None, obs_mean=None, obs_std=None, params=None, checkpoint=None,
                 weights_mlp=None):
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
            try:
                weights_mlp = checkpoint["weights_mlp"]
                self.get_weights_mlp(weights_mlp)
            except:
                pass


        self.params = params
        self.n_neuron = params["n_neuron"]
        self.dt = params["dt"]
        self.dt_env = params["dt_env"]
        self.tau = params["tau"]
        self.baseline_input = params["baseline_input"]
        self.initiation_length = params["initiation_length"]
        self.warmup_steps = params['warmup_steps']
        self.min_reward_init = params['min_reward_init']

        self.x = np.random.rand(self.n_neuron) - 0.5
        self.r = self.activate(self.x)
        self.obs = self.env.reset()
        self.reward = 0

        if type(self.env.action_space) is gym.spaces.box.Box:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = 1
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space

        try:
            self.num_hidden_units = params['num_hidden_units']
            self.make_random_weights_mlp()
            # self.net = torch.nn.Sequential(
            #     torch.nn.Linear(self.obs_dim,self.num_hidden_units),
            #     torch.nn.Tanh(),
            #     torch.nn.Linear(self.num_hidden_units, self.num_hidden_units),
            #     torch.nn.Tanh(),
            #     torch.nn.Linear(self.num_hidden_units, self.action_space),
            #     torch.nn.Tanh()
            # )
        except: pass

        self.done = False
        if weights is not None: self.warmup()

    def set_weights(self, weights):
        self.weights = weights
        self.reset()

    def reset(self):
        self.obs = self.env.reset()
        self.x = np.random.rand(self.n_neuron) - 0.5
        self.warmup()

    def step(self):
        self.step_net()
        action = self.compute_action(self.r[-self.action_dim:])
        self.obs, reward, done, _ = self.env.step(action)

        self.done = done
        self.reward += reward

    def step_net(self, add_obs=True):
        input = self.baseline_input * np.ones(self.n_neuron)
        if add_obs: input[:self.obs_dim] += self.compute_observation()
        for _ in range(round(self.dt_env / self.dt)):
            self.x += self.dt / self.tau * (-self.x + np.matmul(self.weights, self.r) + input)
        self.r = self.activate(self.x)

    def compute_action(self, input):
        if type(self.action_space) is gym.spaces.box.Box:
            action = input * (self.action_space.high - self.action_space.low) + self.action_space.low
        else:
            action = min(int(np.floor(self.action_space.n * input) + self.action_space.start), self.action_space.n - 1)
        return action

    def compute_observation(self):
        return (self.obs - self.obs_mean) / self.obs_std

    def initiation(self):
        print("Initializing ...")
        obs_all = np.zeros((self.initiation_length, self.obs_dim))
        for k in range(self.initiation_length):
            obs, reward, done, info = self.env.step(self.action_space.sample())
            if done:
                self.reset()
            obs_all[k, :] = obs

        self.obs_mean = obs_all.mean(axis=0)
        self.obs_std = obs_all.std(axis=0)
        self.obs_std[self.obs_std==0] = 0.1
        print("obs_mean: ",self.obs_mean,'    obs_std: ',self.obs_std)

        self.reset()
        reward_max = -1e8
        weights_best = None
        reward_total = 0

        k = 0
        while reward_max < self.min_reward_init and k < self.initiation_length:
            k += 1
            self.step_net()
            action = self.compute_action(self.r[-self.action_dim:])
            self.obs, reward, done, _ = self.env.step(action)
            reward_total += reward

            if done:
                if reward_total > reward_max:
                    weights_best = self.weights
                    reward_max = reward_total

                reward_total = 0
                self.reset()
                self.weights = make_random_weights(self.n_neuron)
        print("Obversation mean: ", self.obs_mean)
        print("Obversation std: ", self.obs_std)
        print("Best reward: ", reward_max)
        return (self.obs_mean, self.obs_std, weights_best)

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
        reward_max = -1e8
        weights_best = None
        reward_total = 0

        k = 0
        weights_mlp = self.make_random_weights_mlp()
        while reward_max < self.min_reward_init and k < self.initiation_length:
            k += 1

            obs = self.compute_observation()
            hidden = np.tanh(np.matmul(self.weights_in, obs) + self.bias_in)
            hidden2 = np.tanh(np.matmul(self.weights_hidden, hidden) + self.bias_hidden)
            out = self.activate(np.matmul(self.weights_out, hidden2) + self.bias_out)
            action = self.compute_action(out)
            self.obs, reward, done, _ = self.env.step(action)
            reward_total += reward

            if done:
                if reward_total > reward_max:
                    weights_best = weights_mlp
                    reward_max = reward_total

                reward_total = 0
                self.obs = self.env.reset()
                weights_mlp = self.make_random_weights_mlp()

        print("Obversation mean: ", self.obs_mean)
        print("Obversation std: ", self.obs_std)
        print("Best reward: ", reward_max)
        return (self.obs_mean, self.obs_std, weights_best)

    def warmup(self):
        for _ in range(self.warmup_steps):
            self.step_net(add_obs=False)

    def activate(self, x):
        return expit(x)

    def visualize(self, duration = 1e8):
        self.reset()
        if self.weights is not None:
            print('RateRNN policy!')
            for _ in range(duration):
                self.step_net()
                action = self.compute_action(self.r[-self.action_dim:])
                self.obs, reward, done, _ = self.env.step(action)
                self.env.render()
                time.sleep(0.001)
        else:
            print('MLP policy!')
            for _ in range(duration):
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

    def get_example_trace(self, num=5):
        self.obs = self.env.reset()
        self.x = np.random.rand(self.n_neuron) - 0.5
        idx_neuron = np.random.permutation(self.n_neuron)
        idx_neuron = idx_neuron[:num]
        x_rec = [self.x.copy()]
        for k in range(self.warmup_steps):
            input = self.baseline_input * np.ones(self.n_neuron)
            for _ in range(round(self.dt_env / self.dt)):
                self.x += self.dt / self.tau * (-self.x + np.matmul(self.weights, self.r) + input)
                x_rec.append(self.x.copy())
            self.r = self.activate(self.x)

        done = False
        while not done:
            input = self.baseline_input * np.ones(self.n_neuron)
            input[:self.obs_dim] += self.compute_observation()
            for _ in range(round(self.dt_env / self.dt)):
                self.x += self.dt / self.tau * (-self.x + np.matmul(self.weights, self.r) + input)
                x_rec.append(self.x.copy())
            self.r = self.activate(self.x)
            action = self.compute_action(self.r[-self.action_dim:])
            self.obs, reward, done, _ = self.env.step(action)

        x_rec_np = np.array(x_rec)
        x_rec_np = x_rec_np[:,-self.action_dim:]
        # x_rec_np = x_rec_np[:,idx_neuron]
        r_rec_np = self.activate(x_rec_np)

        plt.subplot(1,2,1)
        plt.title('x')
        plt.plot(np.arange(x_rec_np.shape[0])*self.dt,x_rec_np)
        plt.xlabel('time (s)')
        plt.subplot(1,2,2)
        plt.title('Rate')
        plt.plot(np.arange(x_rec_np.shape[0])*self.dt,r_rec_np)
        plt.xlabel('time (s)')

        plt.show()

    def record_video(self, path='vid.mp4', record_length=1000, width=1024, height=1024, fp=60):
        self.reset()
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(path, fourcc, fp, (width, height), True)  # 最后一个是保存图片的尺寸

        for i in range(record_length):
            self.step_net()
            action = self.compute_action(self.r[-self.action_dim:])
            self.obs, reward, done, _ = self.env.step(action)
            frame = self.env.render(mode='rgb_array', width=width, height=height)
            videoWriter.write(frame)

        videoWriter.release()


    def get_total_reward(self):
        return self.reward

    def get_x(self):
        return self.x

    def get_r(self):
        return self.r

    def step_mlp(self):
        obs = self.compute_observation()
        hidden = np.tanh(np.matmul(self.weights_in, obs) + self.bias_in)
        hidden2 = np.tanh(np.matmul(self.weights_hidden, hidden) + self.bias_hidden)
        out = self.activate(np.matmul(self.weights_out, hidden2) + self.bias_out)
        action = self.compute_action(out)

        self.obs, reward, done, _ = self.env.step(action)
        self.done = done
        self.reward += reward

    def get_weights_mlp(self,weights_mlp):
        p =0
        self.weights_in = np.reshape(weights_mlp[:self.obs_dim * self.num_hidden_units],
                                     (self.num_hidden_units, self.obs_dim))
        p+=self.obs_dim * self.num_hidden_units
        self.bias_in = np.reshape(weights_mlp[
                               p:p + self.num_hidden_units],
                               (self.num_hidden_units,))
        p+=self.num_hidden_units
        self.weights_hidden = np.reshape(weights_mlp[:self.num_hidden_units * self.num_hidden_units],
                                     (self.num_hidden_units, self.num_hidden_units))
        p+=self.num_hidden_units * self.num_hidden_units
        self.bias_hidden = np.reshape(weights_mlp[
                               p:p + self.num_hidden_units],
                               (self.num_hidden_units,))
        p+=self.num_hidden_units
        self.weights_out = np.reshape(weights_mlp[p:p+self.action_dim * self.num_hidden_units],
                                      (self.action_dim, self.num_hidden_units))
        p+=self.action_dim*self.num_hidden_units
        self.bias_out = np.reshape(weights_mlp[p:],(self.action_dim,))

    def make_random_weights_mlp(self):
        self.weights_num = (self.obs_dim+1+self.num_hidden_units+1) * self.num_hidden_units + (self.num_hidden_units+1)*self.action_dim
        weights_mlp = (np.random.rand(self.weights_num) - 0.5) * 2

        self.get_weights_mlp(weights_mlp)
        self.bias_in[:] = 0
        self.bias_hidden[:] = 0
        self.bias_out[:] = 0

        return weights_mlp

    def close(self):
        self.env.close()


