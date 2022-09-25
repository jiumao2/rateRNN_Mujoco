import gym

import utils
from get_hyperparams import get_hyperparams
from rateRNN import rateRNN
import numpy as np
from matplotlib import pyplot as plt
import utils
import pickle

log_dir = r"./results/InvertedDoublePendulum-v4_20220925_200253/"

utils.visualize_log(log_dir+"log.pickle")

data = utils.load_pickle(log_dir + "InvertedDoublePendulum-v4_checkpoint_iter521.pickle")
print(data['params'])
rnn = rateRNN(checkpoint=data)
rnn.get_example_trace()
rnn.visualize_weight()
rnn.visualize(duration=1000)
# rnn.record_video(path=log_dir+"vid.mp4", record_length=1000, width=720, height=720, fp=60)