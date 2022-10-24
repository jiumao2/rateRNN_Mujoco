import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


def compute_centered_ranks(g):
    x = g.reshape((-1,))
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))

    x /= x.size - 1
    x -= 0.5

    g = x.reshape(g.shape)
    return g


def make_random_weights(n_neuron, n_axis3=0):
    if n_axis3 == 0:
        return (np.random.rand(n_neuron, n_neuron) - 0.5) * 2
    else:
        return (np.random.rand(n_neuron, n_neuron, n_axis3) - 0.5) * 2


def make_random_weights_simple(obs_dim, act_dim):
    return (np.random.rand(obs_dim * (act_dim+1)) - 0.5) * 2


def save_checkpoint(dir_name, iter=None, env_name=None, weights=None, params=None, obs_mean=None, obs_std=None,
                    weights_parent=None, weights_children=None):
    result = dict()
    result["iteration"] = iter
    result["env_name"] = env_name
    result["weights"] = weights
    result["params"] = params
    result["obs_mean"] = obs_mean
    result["obs_std"] = obs_std
    result["weights_parent"] = weights_parent
    result["weights_children"] = weights_children
    result["log_dir"] = dir_name

    filename = dir_name + env_name + "_checkpoint_iter" + str(iter) + ".pickle"
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

    print("Save checkpoint at iteration " + str(iter) + ' into "' + filename + '"!')
    return result


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data


def update_log(filename, iter, mean_reward, max_reward, min_reward, mean_length, max_length, min_length):
    names = ["iter", "mean_reward", "max_reward", "min_reward", "mean_episode_length", "max_episode_length",
             "min_episode_length"]
    if not os.path.isfile(filename):
        result = dict()
        result["iter"] = [iter]
        result["mean_reward"] = [mean_reward]
        result["max_reward"] = [max_reward]
        result["min_reward"] = [min_reward]
        result["mean_episode_length"] = [mean_length]
        result["max_episode_length"] = [max_length]
        result["min_episode_length"] = [min_length]
    else:
        result = load_pickle(filename)

        if result["iter"][-1] >= iter:
            for name in names:
                result[name] = result[name][:iter - 1]

        result["iter"].append(iter)
        result["mean_reward"].append(mean_reward)
        result["max_reward"].append(max_reward)
        result["min_reward"].append(min_reward)
        result["mean_episode_length"].append(mean_length)
        result["max_episode_length"].append(max_length)
        result["min_episode_length"].append(min_length)

    with open(filename, 'wb') as f:
        pickle.dump(result, f)

    return result


def visualize_log(filename):
    data = load_pickle(filename)
    # plt.subplot(1,2,1)
    # plt.plot(data['iter'], data['mean_reward'])
    plt.plot(data['iter'], data['max_reward'])
    # plt.plot(data['iter'], data['min_reward'])
    plt.xlabel('Iteration')
    plt.title('Reward')
    # plt.legend(['Mean','Max','Min'],loc=1)

    # plt.subplot(1,2,2)
    # plt.plot(data['iter'], data['mean_episode_length'])
    # # plt.plot(data['iter'], data['max_episode_length'])
    # # plt.plot(data['iter'], data['min_episode_length'])
    # plt.xlabel('Iteration')
    # plt.title('Episode length')
    # plt.legend(['Mean','Max','Min'],loc=1)

    plt.show()
