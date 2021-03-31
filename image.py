# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Action.recognizer import load_action_premodel, framewise_recognize
from Pose.pose_visualizer import TfPoseVisualizer
from utils import choose_run_mode, load_pretrain_model, set_video_writer
import time
import numpy as np
import argparse
import cv2 as cv
from Action.action_enum import Actions


# parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
# parser.add_argument('--video', help='Path to video file.')
# args = parser.parse_args()

# 导入相关模型
estimator = load_pretrain_model('VGG_origin')
# action_classifier = load_action_premodel('Action/framewise_recognition.h5')
action_classifier = load_action_premodel('Action/framewise_recognition_under_scene.h5')

# 参数初始化
# realtime_fps = '0.0000'
# start_time = time.time()
# fps_interval = 1
# fps_count = 0
# run_timer = 0
# frame_count = 0

# 读写视频文件（仅测试过webcam输入）
#cap = choose_run_mode(args)
# video_writer = set_video_writer(cap, write_fps=int(7.0))

# # 保存关节数据的txt文件，用于训练过程(for training)
# f = open('origin_data.txt', 'a+')

# while cv.waitKey(1) < 0:
#    has_frame, show = cap.read()
#    if has_frame:

# fps_count += 1
# frame_count += 1

randimg = 'C:/tmp/10.jpg'
show = cv.imread(randimg)

# pose estimation
humans = estimator.inference(show)
# get pose info and return frame, joints, bboxes, xcenter
pose = TfPoseVisualizer.draw_pose_rgb(show, humans)
frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
print('bboxes', bboxes)
# xcenter是一帧图像中每个human的第1号关节点(neck)的x坐标值
print('xcenter', xcenter)
# 记录每一帧的所有关节点
print('joints', joints)

print('pose[4]', pose[4])
print('pose[4].length', len(pose[4]))

joints_norm_per_frame = np.array(pose[4])
print('joints_norm_per_frame', joints_norm_per_frame)
j = 0
joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
joints_norm_single_person = joints_norm_per_frame
joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
data = action_classifier.predict(joints_norm_single_person)
print('data', data)
pred = np.argmax(data)
print('pred', pred)
print('Actions', Actions)
init_label = Actions(pred).name
print('init_label', init_label)

# recognize the action framewise
show = framewise_recognize(pose, action_classifier)

height, width = show.shape[:2]

cv.putText(show, init_label, (5, height-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# 显示实时FPS值
# if (time.time() - start_time) > fps_interval:
#     # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
#     realtime_fps = fps_count / (time.time() - start_time)
#     fps_count = 0  # 帧数清零
#     start_time = time.time()
# fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
# cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# 显示检测到的人数
num_label = "Human: {0}".format(len(humans))
cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# 显示目前的运行时长及总帧数
# if frame_count == 1:
#     run_timer = time.time()
# run_time = time.time() - run_timer
# time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
# cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

cv.imshow('Action Recognition based on OpenPose', show)
cv.waitKey(0)

# video_writer.write(show)

# # 采集数据，用于训练过程(for training)
# joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
# f.write(' '.join(joints_norm_per_frame))
# f.write('\n')

# video_writer.release()
# cap.release()
# f.close()
