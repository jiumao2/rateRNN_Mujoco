import gym

import utils
from get_hyperparams import get_hyperparams
from rateRNN import rateRNN
import numpy as np
from matplotlib import pyplot as plt
import utils
import pickle
import cv2

log_dir = r"./results/Walker2d-v4_20220923_144247/"

# utils.visualize_log(log_dir+"log.pickle")

# data = utils.load_pickle(log_dir + "Walker2d-v4_checkpoint_iter1001.pickle")
# print(data['params'])
# rnn = rateRNN(checkpoint=data)
# rnn.get_all_trace()
# rnn.pca_plot()
# rnn.get_example_trace()
# rnn.visualize_weight()
# rnn.visualize(duration=1000)
# rnn.close()
iters = [21,41,61,81,101,121,201,1001]
# for iter in iters:
#     data = utils.load_pickle(log_dir + "Walker2d-v4_checkpoint_iter"+str(iter)+".pickle")
#     rnn = rateRNN(checkpoint=data)
#     rnn.record_video(path=log_dir+"vid_iter"+str(iter)+".mp4", record_length=300, width=720, height=720, fp=60)
#     rnn.close()

width = 720
height = 720
fp = 60
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(log_dir+'vid_training.mp4', fourcc, fp, (width, height), True)  # 最后一个是保存图片的尺寸

for iter in iters:
    cap = cv2.VideoCapture(log_dir+'vid_iter'+str(iter)+'.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "Iter "+str(iter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 1)
            videoWriter.write(frame)
        else:
            cap.release()

videoWriter.release()
