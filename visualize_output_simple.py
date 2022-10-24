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

utils.visualize_log(log_dir+"log.pickle")

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
print(data['params'])
print(data['obs_mean'])
print(data['obs_std'])
rnn = Model(checkpoint=data, mask=[3,4,5,6,7])
# rnn.weights_out = np.array([[0,-0.3,-5,0,0,0,0,0],[0,1.9,-5,0,0,0,0,0]])
# rnn.get_all_trace()
# rnn.show_input(duration=200)
# rnn.pca_plot()
# rnn.get_example_trace()
rnn.visualize_weight()
# rnn.visualize(duration=1000)
rnn.show_output(duration=200)
# rnn.record_with_input('vid_simple.mp4', record_length=200, height=1080, width=1080)
rnn.close()
# iters = [21,41,61,81,101,121,201,1001]
# for iter in iters:
#     data = utils.load_pickle(log_dir + "Walker2d-v4_checkpoint_iter"+str(iter)+".pickle")
#     rnn = rateRNN(checkpoint=data)
#     rnn.record_video(path=log_dir+"vid_iter"+str(iter)+".mp4", record_length=300, width=720, height=720, fp=60)
#     rnn.close()

# width = 720
# height = 720
# fp = 60
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# videoWriter = cv2.VideoWriter(log_dir+'vid_training.mp4', fourcc, fp, (width, height), True)  # 最后一个是保存图片的尺寸
#
# for iter in iters:
#     cap = cv2.VideoCapture(log_dir+'vid_iter'+str(iter)+'.mp4')
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             cv2.putText(frame, "Iter "+str(iter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (255, 255, 255), 1)
#             videoWriter.write(frame)
#         else:
#             cap.release()
#
# videoWriter.release()
