import gym
from get_hyperparams import get_hyperparams
from rateRNN import rateRNN
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from utils import compute_centered_ranks, make_random_weights

params = get_hyperparams()
num_workers = params["num_workers"]
num_episodes_per_epoch = params["num_episodes_per_epoch"]
n_neuron = params["n_neuron"]
env_name = "CartPole-v1"
sigma = params['noise_standard_deviation']
lr = params['learning_rate']
episode_length = params["episode_length"]
weights_clip = params["weights_clip"]
l2 = params["l2"]


def process(data):
    gaussian_noise, weights, obs_mean, obs_std = data
    weights_new = (weights + gaussian_noise).clip(-weights_clip, weights_clip)
    rnn = rateRNN(weights_new, env_name, obs_mean=obs_mean, obs_std=obs_std)
    for j in range(episode_length):
        rnn.step()
        length_this = j
        if rnn.done:
            break

    return (rnn.get_total_reward(), length_this)


if __name__ == "__main__":
    pool_obj = multiprocessing.Pool(num_workers)
    env = rateRNN(make_random_weights(n_neuron), env_name)
    obs_mean, obs_std, weights = env.initiation()
    for i in range(1000):
        reward_iter = np.zeros((num_episodes_per_epoch, 2))
        episode_length_iter = np.zeros(num_episodes_per_epoch * 2)
        gaussian_noise = np.random.randn(n_neuron, n_neuron, num_episodes_per_epoch) * sigma
        data_list = []
        for k in range(num_episodes_per_epoch):
            data_list.append((gaussian_noise[:, :, k], weights, obs_mean, obs_std))
            data_list.append((-gaussian_noise[:, :, k], weights, obs_mean, obs_std))

        outcome = pool_obj.map(process, data_list)
        # outcome = []
        # for data in data_list:
        #     outcome.append(process(data))

        for k in range(num_episodes_per_epoch * 2):
            reward_iter[k // 2, k % 2], episode_length_iter[k] = outcome[k]

        print("Mean reward in iteration ", i, ": ", reward_iter.mean())
        print("Max reward: ", reward_iter.max())
        print("Min reward: ", reward_iter.min())
        print("Mean episode length: ", episode_length_iter.mean())
        print("Max episode length: ", episode_length_iter.max())
        print("Min episode length: ", episode_length_iter.min())

        # Update weights
        g = sigma * ((np.array(reward_iter[:, 0]) - np.array(reward_iter[:, 1])) * gaussian_noise).mean(axis=2)
        g = compute_centered_ranks(g)
        g -= l2 * weights
        weights += lr * g
