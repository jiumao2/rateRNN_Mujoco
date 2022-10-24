import time
import matplotlib.pyplot as plt
import numpy as np
import gym
from get_hyperparams import get_hyperparams
from scipy.special import expit
from sklearn.decomposition import PCA
from utils import make_random_weights, make_random_weights_simple
import seaborn as sns


class rateRNN():
    def __init__(self, weights=None,
                 env_name=None,
                 obs_mean=None,
                 obs_std=None,
                 params=None,
                 checkpoint=None,
                 mask=None):
        if checkpoint is None:
            assert env_name is not None
            self.weights = weights
            self.env = gym.make(env_name)

            if params is None:
                params = get_hyperparams()
            else:
                params = params

            self.obs_mean = obs_mean
            self.obs_std = obs_std

        else:
            self.env = gym.make(checkpoint["env_name"])
            params = checkpoint["params"]
            self.obs_mean = checkpoint["obs_mean"]
            self.obs_std = checkpoint["obs_std"]
            self.weights = checkpoint["weights"]

        self.params = params
        self.n_neuron = params["n_neuron"]
        self.dt = params["dt"]
        self.dt_env = params["dt_env"]
        self.tau = params["tau"]
        self.baseline_input = params["baseline_input"]
        self.initiation_length = params["initiation_length"]
        self.warmup_steps = params['warmup_steps']

        self.reset()
        self.reward = 0
        self.done = False

        if type(self.env.action_space) is gym.spaces.box.Box:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = 1
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space

        self.obs_mask = mask

    def set_weights(self, weights):
        self.weights = weights
        self.reset()

    def reset(self):
        self.obs = self.env.reset()
        self.x = np.random.rand(self.n_neuron) - 0.5
        self.r = self.activate(self.x)
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
        temp = (self.obs - self.obs_mean) / self.obs_std
        if self.obs_mask is not None:
            temp[self.obs_mask] = 0
        return temp

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
        self.obs_std[self.obs_std == 0] = 0.1
        print("obs_mean: ", self.obs_mean, '    obs_std: ', self.obs_std)

        self.reset()
        reward_max = -1e8
        weights_best = None
        reward_total = 0

        for k in range(self.initiation_length):
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

    def warmup(self):
        for _ in range(self.warmup_steps):
            self.step_net(add_obs=False)

    def activate(self, x):
        return expit(x)

    def visualize(self, duration=1e8):
        self.reset()
        print('RateRNN policy!')
        for _ in range(duration):
            self.step_net()
            action = self.compute_action(self.r[-self.action_dim:])
            self.obs, reward, done, _ = self.env.step(action)
            self.env.render()
            time.sleep(0.001)

    def visualize_weight(self):
        vmax = np.max(np.abs(self.weights_out), axis=(0, 1))
        plt.imshow(self.weights_out, cmap='bwr', vmax=vmax, vmin=-vmax)
        plt.xlabel('Observation')
        plt.ylabel('Action')
        plt.title('weights')
        plt.colorbar()
        plt.show()
        # plt.hist(self.weights.reshape((-1,)))
        # plt.show()

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
        x_rec_np = x_rec_np[:, -self.action_dim:]
        # x_rec_np = x_rec_np[:,idx_neuron]
        r_rec_np = self.activate(x_rec_np)

        plt.subplot(1, 2, 1)
        plt.title('x')
        plt.plot(np.arange(x_rec_np.shape[0]) * self.dt, x_rec_np)
        plt.xlabel('time (s)')
        plt.subplot(1, 2, 2)
        plt.title('Rate')
        plt.plot(np.arange(x_rec_np.shape[0]) * self.dt, r_rec_np)
        plt.xlabel('time (s)')

        plt.show()

    def get_all_trace(self):
        self.obs = self.env.reset()
        self.x = np.random.rand(self.n_neuron) - 0.5
        idx_neuron = np.random.permutation(self.n_neuron)
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
        r_rec_np = self.activate(x_rec_np)

        plt.subplot(1, 2, 1)
        plt.title('x')
        plt.plot(np.arange(x_rec_np.shape[0]) * self.dt, x_rec_np + np.arange(self.n_neuron) * 5)
        plt.xlabel('time (s)')
        plt.subplot(1, 2, 2)
        plt.title('Rate')
        plt.plot(np.arange(x_rec_np.shape[0]) * self.dt, r_rec_np + np.arange(self.n_neuron) * 1)
        plt.xlabel('time (s)')

        plt.show()

    def show_input(self, duration=1000):
        self.reset()
        obs_rec = []
        for k in range(duration):
            obs_rec.append(self.compute_observation())
            self.step()

        obs_rec = np.array(obs_rec)+np.arange(self.obs_dim)

        plt.figure(figsize=[5,5], dpi=600)
        plt.plot(np.arange(duration), obs_rec)
        plt.xlabel('Steps')
        plt.ylabel('Observation')
        plt.show()

    def record_video(self, path='vid.mp4', record_length=1000, width=1024, height=1024, fp=60):
        self.reset()
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(path, fourcc, fp, (width, height), True)

        for i in range(record_length):
            self.step_net()
            action = self.compute_action(self.r[-self.action_dim:])
            self.obs, reward, done, _ = self.env.step(action)
            frame = self.env.render(mode='rgb_array', width=width, height=height)
            videoWriter.write(frame)

        videoWriter.release()

    def pca_plot(self):
        self.reset()
        x_rec = [self.x.copy()]

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
        r_rec_np = self.activate(x_rec_np)

        pca = PCA(n_components=2)
        pc = pca.fit_transform(x_rec_np)
        print('Explained variance ratio of first two PCs: ', pca.explained_variance_ratio_)

        plt.plot(pc[:, 0], pc[:, 1])
        plt.plot(pc[0, 0], pc[0, 1], 'go')
        plt.plot(pc[-1, 0], pc[-1, 1], 'rx')
        plt.title('PCA')
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        plt.show()

    def get_total_reward(self):
        return self.reward

    def close(self):
        self.env.close()


class simpleRNN(rateRNN):
    def __init__(self, weights=None,
                 env_name=None,
                 obs_mean=None,
                 obs_std=None,
                 params=None,
                 checkpoint=None,
                 mask=None):
        if checkpoint is None:
            assert env_name is not None
            self.weights = weights
            self.env = gym.make(env_name)

            if params is None:
                params = get_hyperparams()
            else:
                params = params

            self.obs_mean = obs_mean
            self.obs_std = obs_std

        else:
            self.env = gym.make(checkpoint["env_name"])
            params = checkpoint["params"]
            self.obs_mean = checkpoint["obs_mean"]
            self.obs_std = checkpoint["obs_std"]
            self.weights = checkpoint["weights"]

        self.params = params
        self.dt = params["dt"]
        self.dt_env = params["dt_env"]
        self.tau = params["tau"]
        self.initiation_length = params["initiation_length"]
        self.warmup_steps = params['warmup_steps']

        self.reward = 0
        self.done = False

        if type(self.env.action_space) is gym.spaces.box.Box:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = 1
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space

        self.obs_mask = mask

        self.n_neuron = self.obs_dim
        if self.weights is None:
            self.weights = make_random_weights_simple(self.obs_dim, self.action_dim)
        self.get_weights()
        self.reset()

    def step_net(self, add_obs=True):
        input = np.zeros(self.n_neuron)
        if add_obs: input += self.compute_observation()
        for _ in range(round(self.dt_env / self.dt)):
            self.x += self.dt / self.tau * (-self.x + self.weights_unit*self.r + input)
        self.r = self.activate(self.x)

    def step(self):
        self.step_net()
        action = self.compute_action(np.matmul(self.weights_out, self.r))
        self.obs, reward, done, _ = self.env.step(action)

        self.done = done
        self.reward += reward

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
        self.obs_std[self.obs_std == 0] = 0.1
        print("obs_mean: ", self.obs_mean, '    obs_std: ', self.obs_std)

        self.reset()
        reward_max = -1e8
        weights_best = None
        reward_total = 0

        for k in range(self.initiation_length):
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
                self.weights = make_random_weights_simple(self.obs_dim, self.action_dim)
                self.get_weights()
        print("Obversation mean: ", self.obs_mean)
        print("Obversation std: ", self.obs_std)
        print("Best reward: ", reward_max)
        return (self.obs_mean, self.obs_std, weights_best)

    def get_weights(self):
        self.weights_unit = self.weights[:self.n_neuron]
        self.weights_out = np.reshape(self.weights[self.n_neuron:], [self.action_dim, self.n_neuron])

    def visualize(self, duration=1e8):
        self.reset()
        print('RateRNN policy!')
        for _ in range(duration):
            self.step_net()
            action = self.compute_action(np.matmul(self.weights_out,self.r))
            self.obs, reward, done, _ = self.env.step(action)
            self.env.render()
            time.sleep(0.001)

class simplestRNN(simpleRNN):
    def get_weights(self):
        self.weights_unit = np.zeros(self.n_neuron)
        self.weights_out = np.reshape(self.weights[self.n_neuron:], [self.action_dim, self.n_neuron])

class nonRNN(simpleRNN):
    def step_net(self, add_obs=True):
        input = np.zeros(self.n_neuron)
        if add_obs: input += self.compute_observation()
        self.x = input
        self.r = self.activate(self.x)

    def visualize_weight(self):
        vmax = np.max(np.abs(self.weights_out), axis=(0, 1))
        plt.imshow(self.weights_out, cmap='bwr', vmax=vmax, vmin=-vmax)
        plt.title('weights')
        plt.colorbar()
        plt.show()
        plt.hist(self.weights_out.reshape((-1,)))
        plt.show()

    def get_all_trace(self):
        self.obs = self.env.reset()
        x_rec = []

        done = False
        while not done:
            input = self.compute_observation()
            self.x = input
            x_rec.append(self.x.copy())
            self.r = self.activate(self.x)
            action = self.compute_action(np.matmul(self.weights_out, self.r))
            self.obs, reward, done, _ = self.env.step(action)

        x_rec_np = np.array(x_rec)
        r_rec_np = self.activate(x_rec_np)

        plt.subplot(1, 2, 1)
        plt.title('x')
        plt.plot(np.arange(x_rec_np.shape[0]) * self.dt, x_rec_np + np.arange(self.n_neuron) * 5)
        plt.xlabel('time (s)')
        plt.subplot(1, 2, 2)
        plt.title('Rate')
        plt.plot(np.arange(x_rec_np.shape[0]) * self.dt, r_rec_np + np.arange(self.n_neuron) * 1)
        plt.xlabel('time (s)')

        plt.show()

class nonNN(simpleRNN):
    def step_net(self, add_obs=True):
        input = np.zeros(self.n_neuron)
        if add_obs: input += self.compute_observation()
        self.x = input
        self.r = self.x

    def get_all_trace(self):
        self.obs = self.env.reset()
        x_rec = []

        done = False
        while not done:
            input = self.compute_observation()
            self.x = input
            x_rec.append(self.x.copy())
            self.r = self.x
            action = self.compute_action(np.matmul(self.weights_out, self.r))
            self.obs, reward, done, _ = self.env.step(action)

        x_rec_np = np.array(x_rec)
        r_rec_np = self.activate(x_rec_np)

        plt.subplot(1, 2, 1)
        plt.title('x')
        plt.plot(np.arange(x_rec_np.shape[0]) * self.dt, x_rec_np + np.arange(self.n_neuron) * 5)
        plt.xlabel('time (s)')
        plt.subplot(1, 2, 2)
        plt.title('Rate')
        plt.plot(np.arange(x_rec_np.shape[0]) * self.dt, r_rec_np + np.arange(self.n_neuron) * 1)
        plt.xlabel('time (s)')

        plt.show()

    def record_with_input(self, path='vid.mp4', record_length=1000, width=720, height=720, fp=60):
        self.reset()
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = None
        obs_rec = []
        act_rec = []

        for i in range(record_length):
            self.step_net()
            action = self.compute_action(np.matmul(self.weights_out, self.r))
            action = action.clip(-1, 1)
            act_rec.append(action)
            self.obs, reward, done, _ = self.env.step(action)
            obs_rec.append(self.compute_observation()[:3])
            frame = self.env.render(mode='rgb_array', width=width, height=height)

            fig = plt.figure(figsize=[5,5], dpi=600)
            plt.plot(obs_rec)
            plt.plot(act_rec)
            plt.xlim([0,record_length])
            plt.ylim([-2,2])
            plt.xlabel('Steps')
            plt.ylabel('Observation')
            plt.legend(['0','1','2','action 0','action 1'])
            plt.plot([0,record_length],[0,0])
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            width_plot, height_plot, _ = data.shape
            img_plot = cv2.resize(data, (int(width_plot*height/height_plot), height))
            frame = np.concatenate((frame,img_plot), axis=1)

            plt.close(fig)

            if videoWriter is None:
                videoWriter = cv2.VideoWriter(path, fourcc, fp, (frame.shape[1], frame.shape[0]), True)
            videoWriter.write(frame)
            print(i)

        videoWriter.release()

    def show_output(self, duration=1000):
        self.reset()
        act_rec = []
        for k in range(duration):
            self.step_net()
            action = self.compute_action(np.matmul(self.weights_out, self.r))
            action = np.array(action).clip(-1,1)
            act_rec.append(action)
            self.obs, reward, done, _ = self.env.step(action)

            self.done = done
            self.reward += reward

        act_rec = np.array(act_rec) + np.arange(self.action_dim)

        plt.figure(figsize=[5,5], dpi=600)
        plt.plot(np.arange(duration), act_rec)
        plt.xlabel('Steps')
        plt.ylabel('Action')
        plt.yticks([0,1])
        plt.show()