import os
from get_hyperparams import get_hyperparams
import numpy as np
import multiprocessing
import utils
import datetime
import warnings
from MLP import MLP

warnings.filterwarnings("ignore")

params = get_hyperparams()
num_workers = params["num_workers"]
env_name = "CartPole-v1"
episode_length = params["episode_length"]
weights_clip = params["weights_clip"]
N = params['population_size']
sigma = params['mutation_power']
T = params['truncation_size']


def process(data, times=5):
    weights, obs_mean, obs_std = data
    weights_new = weights.clip(-weights_clip, weights_clip)
    rnn = MLP(env_name=env_name, obs_mean=obs_mean, obs_std=obs_std, weights=weights_new)
    reward = 0
    for _ in range(times):
        for j in range(episode_length):
            rnn.step()
            length_this = j
            if rnn.done:
                break
        reward += rnn.get_total_reward()

    return (reward/times, length_this)


if __name__ == "__main__":
    pool_obj = multiprocessing.Pool(4)
    Elite = []
    continue_training = False
    if continue_training:
        checkpoint = utils.load_pickle('./results/Ant-v4_20220917_135626/Ant-v4_checkpoint_iter501.pickle')
        log_dir = checkpoint['log_dir']
        obs_mean = checkpoint['obs_mean']
        obs_std = checkpoint['obs_std']
        weights_parent = checkpoint['weights_parent']
        iter_this = checkpoint['iteration']
        weights_size = np.size(weights_parent[0])
    else:
        log_dir = 'results/' + env_name + '_GA_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
        os.mkdir(log_dir)
        env = MLP(env_name=env_name)
        obs_mean, obs_std, weights = env.initiation_mlp()
        weights_size = np.size(weights)
        iter_this = 0

    log_filename = log_dir + 'log.pickle'

    for i in range(iter_this, 100000):
        if i>1000:
            sigma = 0.001
        if i == 0:
            weights_children = [
                env.net.initialize() for _ in range(N)]
        else:
            gaussian_noise = np.random.randn(weights_size, N) * sigma
            weights_children = [
                gaussian_noise[:, k] + weights_parent[int(np.floor(k / (N + 1) * (T - 1e-8)))] for k
                in range(N)]
            weights_children += Elite

        data_list = []
        for k in range(N):
            data_list.append((weights_children[k], obs_mean, obs_std))

        outcome = pool_obj.map(process, data_list)
        # outcome = []
        # for data in data_list:
        #     outcome.append(process(data, 5))

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
        weights_parent = [weights_children[idx] for idx in reward_iter.argsort()[-T:]]

        weights_candidate = [weights_children[idx] for idx in reward_iter.argsort()[-10:]]
        reward_candidate = [reward_iter[idx] for idx in reward_iter.argsort()[-10:]]
        print(reward_candidate)
        candidate = weights_candidate+Elite
        reward = []
        for each in candidate:
            reward_mean, _ = process((each, obs_mean, obs_std), 30)
            reward.append(reward_mean)

        Elite = [candidate[np.argmax(reward)]]
        if i > 10: weights_parent += Elite

        utils.update_log(log_filename, i, reward_iter.mean(), reward_iter.max(), reward_iter.min(),
                         episode_length_iter.mean(), episode_length_iter.max(), episode_length_iter.min())
        if i % 20 == 1:
            utils.save_checkpoint(log_dir, i, env_name, Elite[0], params, obs_mean, obs_std, weights_parent,
                                  weights_children)

    utils.save_checkpoint(log_dir, i, env_name, Elite[0], params, obs_mean, obs_std, weights_parent, weights_children)
