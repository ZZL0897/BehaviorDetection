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
from STfeature_modules import create_ST_image_cupy, create_ST_image_ch2
from STfeature_modules import load_keypoints_info, select_keypoints_ROI
import time
import cupy as cp
from pathlib import Path

# 根据deeplabcut检测的关键点数据和生成的标签提取信息文件，提取RGB和ST roi

roi = 350
newSize = [1920, 1920]
TimeWindow = 8
gpu = True

file_folder = r'F:\recode' # 输入单个视频或者视频文件夹
keypoints_base_folder = r'G:\test'
save_path = r''

video_file_list = []
if os.path.isdir(file_folder):
    video_file = os.listdir(file_folder)
    [video_file_list.append(os.path.join(file_folder, file)) for file in video_file]
elif os.path.isfile(file_folder):
    video_file_list.append(file_folder)

for video in video_file_list:
    video_name = str(Path(video).stem)
    save_folder = os.path.join(save_path, video_name)
    isExists=os.path.exists(save_folder)

    if isExists:
        print('该视频时空图像已生成')

    if not isExists:
        # os.makedirs(save_folder)
        # os.makedirs(os.path.join(save_folder, 'rgb'))
        # os.makedirs(os.path.join(save_folder, 'st'))
        # os.makedirs(os.path.join(save_folder, 'rgb', 'front'))
        # os.makedirs(os.path.join(save_folder, 'rgb', 'posterior'))
        # os.makedirs(os.path.join(save_folder, 'st', 'front'))
        # os.makedirs(os.path.join(save_folder, 'st', 'posterior'))
        cap = cv2.VideoCapture(video)
        maxframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bodyparts, df_x, df_y, df_likelihood, cropping, cropping_parameters = load_keypoints_info(keypoints_base_folder,
                                                                                                  video)
        startFrame = 0
        endFrame = maxframe

        for frameInd in range(startFrame, endFrame, 1):
            if frameInd < maxframe and frameInd < df_x.shape[1]:
                j = 0
                rgbRec = np.empty((roi, roi, 3, len(bodyparts), TimeWindow))
                # Preallocation: 使用cupy时所有数据都需要迁移到gpu上
                if gpu is None:
                    cfrRec = np.empty((roi, roi, TimeWindow, len(bodyparts)))  # 构造裁剪帧帧堆叠三维矩阵
                elif gpu is not None:
                    cfrRec = cp.empty((roi, roi, TimeWindow, len(bodyparts)))

                # Read frames one by one from startFrame to endFrame:
                for fid in range(frameInd + 1 - TimeWindow, frameInd + 1, 1):

                    cap.set(1, fid)
                    ret, frame = cap.read()
                    # cv2.imshow('frame', cv2.resize(frame, (960, 540)))
                    # print(frameInd)
                    ROI_arr = select_keypoints_ROI(bodyparts, df_x, df_y, frameInd, frame, roi, cropping,
                                                   cropping_parameters, gpu)
                    # ROI_arr shape: (roi, roi, 3, len(bodyparts))
                    rgbRec[..., j] = ROI_arr
                    # rgbRec shape: (roi, roi, 3, len(bodyparts), TimeWindow)
                    if gpu is None:
                        ROI_arr_norm = ROI_arr
                        ROI_arr_gray = np.empty((roi, roi, len(bodyparts)))
                    elif gpu is not None:
                        ROI_arr_norm = cp.asnumpy(ROI_arr)
                        ROI_arr_gray = np.empty((roi, roi, len(bodyparts)))
                    # print(ROI_arr_gray.shape)
                    # Check frames and convert to grayscale:
                    if np.size(np.shape(ROI_arr_norm)) >= 2:
                        for nb in range(0, len(bodyparts)):
                            gray = ROI_arr_norm[..., nb].astype(np.float32)
                            ROI_arr_gray[..., nb] = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
                        if gpu is not None:
                            ROI_arr_gray = cp.array(ROI_arr_gray).astype(float)
                    else:
                        print('Corrupt frame with less than 2 dimensions!!')
                        if gpu is None:
                            ROI_arr_gray = np.zeros((roi, roi, len(bodyparts)))  # Fill the corrupt frame with black
                        elif gpu is not None:
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
                            #     cv2.waitKey(0)
                            im3C = create_ST_image_ch2(cfrRec_clip.astype('float32'), rgbRec_clip.astype('float32'),
                                                           TimeWindow, roi)
                            im3C_arr[..., nb] = im3C
                        image_name = str(frameInd) + '.png'
                        # cv2.imencode('.png', cv2.resize(ROI_arr[..., 0], (roi, roi)))[1].tofile(
                        #     os.path.join(save_folder, 'rgb', 'front', image_name))
                        # cv2.imencode('.png', cv2.resize(ROI_arr[..., 1], (roi, roi)))[1].tofile(
                        #     os.path.join(save_folder, 'rgb', 'posterior', image_name))
                        # cv2.imencode('.png', cv2.resize(im3C_arr[..., 0], (roi, roi)))[1].tofile(
                        #     os.path.join(save_folder, 'st', 'front', image_name))
                        # cv2.imencode('.png', cv2.resize(im3C_arr[..., 1], (roi, roi)))[1].tofile(
                        #     os.path.join(save_folder, 'st', 'posterior', image_name))

                cv2.waitKey(1)

# Close all opencv stuff
cap.release()
cv2.destroyAllWindows()
