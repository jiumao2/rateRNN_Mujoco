import os
from get_hyperparams import get_hyperparams
from rateRNN import rateRNN
import numpy as np
import multiprocessing
import utils
import datetime

continue_training = False
checkpoint_path = r'results/Swimmer-v4_20220927_154629/Swimmer-v4_checkpoint_iter981.pickle'

params = get_hyperparams()
num_workers = params["num_workers"]
n_neuron = params["n_neuron"]
env_name = "Swimmer-v4"
sigma = params['noise_standard_deviation']
weights_clip = params["weights_clip"]
num_parents = params['num_parents']
num_children = params['num_children']
l1 = params['l1']


def process(data):
    weights, obs_mean, obs_std = data
    weights_new = weights.clip(-weights_clip, weights_clip)
    total_reward = 0
    total_length = 0
    times = 1
    for k in range(times):
        rnn = rateRNN(weights_new, env_name, obs_mean=obs_mean, obs_std=obs_std)
        while not rnn.done:
            rnn.step()
            total_length += 1
            if rnn.done:
                break

        rnn.close()
        total_reward += rnn.get_total_reward() - l1*np.sum(np.abs(weights_new))

    return total_reward / times, total_length / times


if __name__ == "__main__":
    pool_obj = multiprocessing.Pool(num_workers)

    if continue_training:
        checkpoint = utils.load_pickle(checkpoint_path)
        env_name = checkpoint['env_name']
        log_dir = checkpoint['log_dir']
        obs_mean = checkpoint['obs_mean']
        obs_std = checkpoint['obs_std']
        weights_parent = checkpoint['weights_parent']
        weights_children = checkpoint['weights_parent']
        num_parents = len(weights_parent)
        iter_this = checkpoint['iteration'] + 1
    else:
        log_dir = 'results/' + env_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
        os.mkdir(log_dir)
        env = rateRNN(utils.make_random_weights(n_neuron), env_name)
        obs_mean, obs_std, weights = env.initiation()
        weights_parent = [utils.make_random_weights(n_neuron) for _ in range(num_parents)]
        weights_parent[0] = weights
        iter_this = 0
        env.close()

    log_filename = log_dir + 'log.pickle'

    for i in range(iter_this, 100000):
        gaussian_noise = np.random.randn(n_neuron, n_neuron, num_children) * sigma
        gaussian_noise[:, :, 1::4] *= 0.5
        gaussian_noise[:, :, 2::4] *= 0.1
        gaussian_noise[:, :, 3::4] *= 0.05
        gaussian_noise[:, :, ::4] *= 0.01
        data_list = []

        weights_children = [
            gaussian_noise[:, :, k] + weights_parent[int(np.floor(k / (num_children + 1) * (num_parents - 1e-8)))] for k
            in range(num_children)]
        weights_children += weights_parent

        for k in range(len(weights_children)):
            data_list.append((weights_children[k], obs_mean, obs_std))

        outcome = pool_obj.map(process, data_list)
        # outcome = []
        # for data in data_list:
        #     outcome.append(process(data))

        reward_iter = np.zeros(len(outcome))
        episode_length_iter = np.zeros(len(outcome))
        for k in range(len(outcome)):
            reward_iter[k], episode_length_iter[k] = outcome[k]

        print("Mean reward in iteration ", i, ": ", reward_iter.mean())
        print("Max reward: ", reward_iter.max())
        print("Min reward: ", reward_iter.min())
        print("Mean episode length: ", episode_length_iter.mean())
        print("Max episode length: ", episode_length_iter.max())
        print("Min episode length: ", episode_length_iter.min())

        # Update weights
        weights_parent = [weights_children[idx] for idx in reward_iter.argsort()[-num_parents:]]
        reward_parent = [reward_iter[idx] for idx in reward_iter.argsort()[-num_parents:]]
        print(reward_parent)

        utils.update_log(log_filename, i, reward_iter.mean(), reward_iter.max(), reward_iter.min(),
                         episode_length_iter.mean(), episode_length_iter.max(), episode_length_iter.min())
        if i % 20 == 1:
            utils.save_checkpoint(log_dir, i, env_name, weights_parent[-1], params, obs_mean, obs_std, weights_parent,
                                  weights_children)

    utils.save_checkpoint(log_dir, i, env_name, weights_parent[-1], params, obs_mean, obs_std, weights_parent,
                          weights_children)
