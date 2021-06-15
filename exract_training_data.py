import pandas as pd
import os
import re
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter
from tkinter import filedialog
import STfeature_modules
from STfeature_modules import create_ST_image_ch2
from STfeature_modules import load_keypoints_info, select_keypoints_ROI, frame_crop
import time
import cupy as cp
from pathlib import Path

# 根据deeplabcut检测的关键点数据和生成的标签提取信息文件，提取RGB和ST roi

roi = 160
newSize = [1920, 1920]
TimeWindow = 8
gpu = False

file_folder = r'F:\recode'
keypoints_base_folder = r'F:\detect'

video_file_list = os.listdir(file_folder)

random_file = pd.read_csv('random.csv')
video_id_list = random_file['视频']
# video_id_list = int(video_id_list)
motion_list = random_file['行为']
frame_id_list = random_file['帧']

vid = str(0)
for i in range(0, len(video_id_list)):
    if len(str(int(video_id_list[i]))) == 2:
        video_name = '000' + str(int(video_id_list[i]))
    elif len(str(int(video_id_list[i]))) == 3:
        video_name = '00' + str(int(video_id_list[i]))

    if int(vid) != int(video_name):
        vid = video_name
        print(vid)
        for video_file in video_file_list:
            if Path(video_file).stem == video_name:
                video_path = file_folder + '/' + video_file
                cap = cv2.VideoCapture(video_path)
                maxframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                bodyparts, df_x, df_y, df_likelihood, cropping, cropping_parameters = load_keypoints_info(keypoints_base_folder,
                                                                                                          video_path)
    frameInd = frame_id_list[i]
    startFrame = frame_id_list[i] + 1 - TimeWindow
    endFrame = frame_id_list[i] + 1

    if frame_id_list[i] <= maxframe and frame_id_list[i] <= df_x.shape[1]:
        j = 0
        # Preallocation: 使用cupy时所有数据都需要迁移到gpu上
        if not gpu:
            cfrRec = np.empty((roi, roi, TimeWindow, len(bodyparts)))  # 构造裁剪帧帧堆叠三维矩阵
            rgbRec = np.empty((roi, roi, 3, len(bodyparts), TimeWindow))
        elif gpu:
            cfrRec = cp.empty((roi, roi, TimeWindow, len(bodyparts)))
            rgbRec = cp.empty((roi, roi, 3, len(bodyparts), TimeWindow))

        # Read frames one by one from startFrame to endFrame:
        for fid in range(startFrame, endFrame, 1):

            cap.set(1, fid)
            ret, frame = cap.read()
            # cv2.imshow('frame', cv2.resize(frame, (960, 540)))
            # print(frameInd)
            frame_cropped = frame_crop(frame, cropping, cropping_parameters)
            ROI_arr = select_keypoints_ROI(bodyparts, df_x, df_y, frameInd, frame_cropped, roi, gpu)
            # ROI_arr shape: (roi, roi, 3, len(bodyparts))
            rgbRec[..., j] = ROI_arr
            if not gpu:
                ROI_arr_norm = ROI_arr
                ROI_arr_gray = np.empty((roi, roi, len(bodyparts)))
            elif gpu:
                ROI_arr_norm = cp.asnumpy(ROI_arr)
                ROI_arr_gray = np.empty((roi, roi, len(bodyparts)))
            # print(ROI_arr_gray.shape)
            # Check frames and convert to grayscale:
            if np.size(np.shape(ROI_arr_norm)) >= 2:
                for nb in range(0, len(bodyparts)):
                    gray = ROI_arr_norm[..., nb].astype(np.float32)
                    ROI_arr_gray[..., nb] = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
                if gpu:
                    ROI_arr_gray = cp.array(ROI_arr_gray).astype(float)
            else:
                print('Corrupt frame with less than 2 dimensions!!')
                if not gpu:
                    ROI_arr_gray = np.zeros((roi, roi, len(bodyparts)))  # Fill the corrupt frame with black
                elif gpu:
                    ROI_arr_gray = cp.zeros((roi, roi, len(bodyparts)))

            for nb in range(0, len(bodyparts)):
                cfrRec[..., j, nb] = ROI_arr_gray[..., nb]
            j += 1

            if fid == frameInd:
                rgbRecswap = rgbRec.swapaxes(3, 4)
                im3C_arr = np.empty((roi, roi, 3, len(bodyparts)))
                for nb in range(0, len(bodyparts)):
                    cfrRec_clip = cfrRec[..., nb]
                    rgbRec_clip = rgbRecswap[..., nb]
                    # cv2.imshow('t', tfrRec_clip[..., 5])
                    # cv2.waitKey(0)
                    # for k in range(0, TimeWindow):
                    #     cv2.imshow('k', cp.asnumpy(cfrRec_clip[..., k]))
                    #      cv2.waitKey(0)
                    im3C = create_ST_image_ch2(cfrRec_clip.astype('float32'), rgbRec_clip.astype('float32'),
                                               TimeWindow, roi, gpu)
                    im3C_arr[..., nb] = im3C

                STfeature_modules.save_img_data(path=r'F:', file_name=str(vid) + '_' + str(frameInd),
                                                bodyparts=bodyparts, RGB_arr=ROI_arr, ST_arr=im3C_arr, save_size=roi,
                                                video_id=vid, label_name=str(motion_list[i]))

        print(motion_list[i])

        cv2.waitKey(1)

# Close all opencv stuff
cap.release()
cv2.destroyAllWindows()
