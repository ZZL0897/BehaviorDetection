import numpy as np
import cv2
import cupy as cp
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import os
import json
import pandas as pd
import re
import random as rd
import shutil
from pathlib import Path
import pickle

def show_files(path, all_files, all_files_path):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
    # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files, all_files_path)
        else:
            all_files.append(str(Path(file).stem))
            all_files_path.append(cur_path)

    return all_files, all_files_path

# 根据人工检查过的ST图片，把正确样本视频和标签提取出来
def data_clean():
    clean_image_list = os.listdir(r'G:\st\STimg\right') # 输入清理后的ST图像路径
    x_list = os.listdir(r'G:\st\x') # 输入还未清理的TrainData和Label的路径
    y_list = os.listdir(r'G:\st\y')
    for clean_image in clean_image_list:
        clean_image_name = clean_image[:-4]
        shutil.copy(r'G:\st\x/' + clean_image_name + '.avi', r'G:\st\right_x')
        shutil.copy(r'G:\st\y/' + clean_image_name + '.json', r'G:\st\right_y')

# 把之前的结果进行汇总，并把没区分开的行为区间单独提取出来
def merge_result(path=r'C:\Users\ZZL\Desktop\桔小实蝇\res', savepath=r'C:\Users\ZZL\Desktop\桔小实蝇'):
    file_name_list = os.listdir(path)

    video_id_list = []
    start_list = []
    end_list = []
    time_list = []
    motion_list = []
    video_id_list_mis = []
    start_mis = []
    end_mis = []
    time_mis = []
    motion_mis = []

    for f in range(0, len(file_name_list)):
        result_file = pd.read_csv(path + '/' + file_name_list[f])

        video_id = file_name_list[f][2:5]
        print(video_id)
        start = result_file['开始帧']
        end = result_file['结束帧']
        motion = result_file['行为']

        for m in range(0, len(motion)):
            if re.search('/', str(motion[m])):
                video_id_list_mis.append(video_id)
                start_mis.append(start[m])
                end_mis.append(end[m])
                time_mis.append(end[m]-start[m])
                motion_mis.append(motion[m])
            else:
                video_id_list.append(video_id)
                start_list.append(start[m])
                end_list.append(end[m])
                time_list.append(end[m]-start[m])
                motion_list.append(motion[m])

    merge = pd.DataFrame(columns=['视频', '行为', '开始帧', '结束帧', '持续帧数'])
    merge_mis = pd.DataFrame(columns=['视频', '行为', '开始帧', '结束帧', '持续帧数'])
    for i in range(0, len(motion_list)):
        merge = merge.append(pd.DataFrame({'视频': [video_id_list[i]], '行为': [motion_list[i]], '开始帧': [start_list[i]],
                                           '结束帧': [end_list[i]], '持续帧数': [time_list[i]]}))

    for i in range(0, len(motion_mis)):
        merge_mis = merge_mis.append(pd.DataFrame({'视频': [video_id_list_mis[i]], '行为': [motion_mis[i]], '开始帧': [start_mis[i]],
                                                   '结束帧': [end_mis[i]], '持续帧数': [time_mis[i]]}))
    merge.to_csv(savepath + '/merge.csv', index=False)
    merge_mis.to_csv(savepath + '/merge_mis.csv', index=False)
# merge_result()

# 保存RGB图像与ST图像数据，每个keypoint都有有一张RGB+一张ST图像，自动分文件夹保存数据，json内保存标签信息
def save_img_data(path, file_name, bodyparts, RGB_arr, ST_arr, save_size, video_id=None, label_name=None):
    labelname_dict = {'0': 0, '头部': 1, '前足': 2, '前中足': 3, '后中足': 4,
                      '后足': 5, '腹部': 6, '翅膀': 7}
    # labelname_dict = {'0': 0, 'head': 1, 'foreleg': 2, 'front-mid': 3, 'hind-mid': 4,
    #                   'hindleg': 5, 'abdomen': 6, 'wing': 7}
    dictlabel = {'video_id': None,
                 'label_name': None,
                 'label': None}
    if video_id:
        dictlabel['video_id'] = int(video_id)
    if label_name:
        dictlabel['label_name'] = label_name
        dictlabel['label'] = int(labelname_dict[str(label_name)])
        file_name = file_name + '_' + str(labelname_dict[str(label_name)])
        OutputPath = path + '/label/' + file_name

    if str(label_name) == '0':
        front_RGB_img_path = path + '/RGB/front/' + str(label_name) + '/' + file_name + '0.png'
        front_ST_img_path = path + '/ST/front/' + str(label_name) + '/' + file_name + '0.png'
        posterior_RGB_img_path = path + '/RGB/posterior/' + str(label_name) + '/' + file_name + '1.png'
        posterior_ST_img_path = path + '/ST/posterior/' + str(label_name) + '/' + file_name + '1.png'
        cv2.imencode('.png', cv2.resize(RGB_arr[..., 0], (save_size, save_size)))[1].tofile(front_RGB_img_path)
        cv2.imencode('.png', cv2.resize(ST_arr[..., 0], (save_size, save_size)))[1].tofile(front_ST_img_path)
        cv2.imencode('.png', cv2.resize(RGB_arr[..., 1], (save_size, save_size)))[1].tofile(posterior_RGB_img_path)
        cv2.imencode('.png', cv2.resize(ST_arr[..., 1], (save_size, save_size)))[1].tofile(posterior_ST_img_path)

    elif label_name in ['0', '头部', '前足', '前中足']:
        front_RGB_img_path = path + '/RGB/front/' + str(label_name) + '/' + file_name + '.png'
        front_ST_img_path = path + '/ST/front/' + str(label_name) + '/' + file_name + '.png'
        cv2.imencode('.png', cv2.resize(RGB_arr[..., 0], (save_size, save_size)))[1].tofile(front_RGB_img_path)
        cv2.imencode('.png', cv2.resize(ST_arr[..., 0], (save_size, save_size)))[1].tofile(front_ST_img_path)

    elif label_name in ['0', '后足', '后中足', '后足', '腹部', '翅膀', '产卵器']:
        posterior_RGB_img_path = path + '/RGB/posterior/' + str(label_name) + '/'+ file_name + '.png'
        posterior_ST_img_path = path + '/ST/posterior/' + str(label_name) + '/' + file_name + '.png'
        cv2.imencode('.png', cv2.resize(RGB_arr[..., 1], (save_size, save_size)))[1].tofile(posterior_RGB_img_path)
        cv2.imencode('.png', cv2.resize(ST_arr[..., 1], (save_size, save_size)))[1].tofile(posterior_ST_img_path)
    '''
    if label_name in ['0', '头部', '前足', '前中足']:
        for nb in range(0, len(bodyparts)):
            img_path = path + '/front/' + str(label_name) + '/' + file_name + '.png'
            cv2.imencode('.png', cv2.resize(RGB_arr[..., nb], (save_size, save_size)))[1].tofile(img_path)
            cv2.imencode('.png', cv2.resize(ST_arr[..., nb], (save_size, save_size)))[1].tofile(img_path)
    elif label_name in ['0', '后足', '后中足', '后足', '腹部', '翅膀']:
        for nb in range(0, len(bodyparts)):
            img_path = path + '/posterior/' + str(label_name) + '/' + file_name + '.png'
            cv2.imencode('.png', cv2.resize(RGB_arr[..., nb], (save_size, save_size)))[1].tofile(img_path)
            cv2.imencode('.png', cv2.resize(ST_arr[..., nb], (save_size, save_size)))[1].tofile(img_path)
    '''
    # print(img_path)
    with open(OutputPath + '.json', 'w+') as f:
        json.dump(dictlabel, f)

def load_keypoints_info(info_path, video_file_name):
    '''
    根据输入的视频文件名称，在info_path内寻找相应的关键点检测信息（关键点、meatadata），
    载入关键点检测信息，并将置信率较低的点的坐标调整为上一帧该点的坐标
    :param path:
    :return:
    bodyparts: index object, shape: (number of parts, )
    df_x, df_y, df_likelihood, numpy array, shape: (number of parts, number of frame)
    cropping_parameters = [x1, x2, y1, y2] 检测时的裁剪区域
    '''
    video_name = str(Path(video_file_name).stem) # 获取视频去掉文件后缀的名称
    # print(len(video_name))
    # print(video_name)
    # path下含有analyze，metadata两个文件夹存放检测数据
    analyze_path = os.path.join(info_path, 'analyze')
    analyze_list = os.listdir(analyze_path)
    meta_path = os.path.join(info_path, 'metadata')
    meta_list = os.listdir(meta_path)
    # 遍历analyze中的文件
    for an_name in analyze_list:
        if re.match(video_name, an_name): # 正则匹配与视频名称对应的检测文件，载入关键点检测数据
            df = pd.read_hdf(os.path.join(analyze_path, an_name))
            nframes = len(df.index)
            bodyparts = df.columns.get_level_values("bodyparts")[::3]
            nparts = len(bodyparts)
            df_x, df_y, df_likelihood = df.values.reshape((nframes, -1, 3)).T
    # 遍历metadata中的文件
    for meta_name in meta_list: # 正则匹配对应的metadata文件，获取检测时的裁剪参数
        if re.match(video_name, meta_name):
            with open(os.path.join(meta_path, meta_name), "rb") as handle:
                metadata = pickle.load(handle)
            cropping = metadata["data"]["cropping"]
            [x1, x2, y1, y2] = metadata["data"]["cropping_parameters"]
            cropping_parameters = [x1, x2, y1, y2]
            # print(cropping)
            # print(x1, x2, y1, y2)
    # 置信率较低的点的坐标调整为上一帧该点的坐标
    for n in range(0, nparts):
        for i in range(1, nframes):
            if df_likelihood[n, i] < 0.5:
                df_x[n, i] = df_x[n, i - 1]
                df_y[n, i] = df_y[n, i - 1]
                df_likelihood[n, i] = 1.0
                if df_likelihood[n, i - 1] < 0.5: # 连续数帧置信率低直接跳过
                    continue
    # print(bodyparts.shape)
    # print(df_x.shape)
    return bodyparts, df_x, df_y, df_likelihood, cropping, cropping_parameters

def frame_crop(frame, cropping, cropping_parameters):
    if cropping:
        [x1, x2, y1, y2] = cropping_parameters
        frame_cropped = frame[y1: y2, x1: x2]
    else:
        frame_cropped = frame

    return frame_cropped

def select_keypoints_ROI(bodyparts, df_x, df_y, frameInd, frame_cropped, roi, gpu=False):
    '''

    :param bodyparts: 需要检测的关键点名称和数量
    :param df_x: 裁剪的x坐标集合，numpy array, shape: (number of parts, number of frame)
    :param df_y: 裁剪的y坐标集合，numpy array, shape: (number of parts, number of frame)
    :param frameInd: 被裁剪帧的帧索引
    :param frame_cropped: 需要裁剪的帧（已经过一次裁剪）
    :param roi: 裁剪区域大小，int
    :param gpu: 是否使用cupy
    :param cropping, cropping_parameters: 裁剪参数
    :return: 返回裁剪后每个bodyparts的ROI，size : (roi, roi, 3, number of parts)
    '''

    frame_cropped_size = frame_cropped.shape
    width = frame_cropped_size[1]
    height = frame_cropped_size[0]
    nparts = len(bodyparts)
    if gpu:
        cp_frame = cp.array(frame_cropped).astype(float)
        cfr_normal = cp.empty((roi, roi, 3))
        ROI_array = cp.empty((roi, roi, 3, nparts))
    elif not gpu:
        ROI_array = np.empty((roi, roi, 3, nparts))
    bottomOvershot = 0
    rightOvershot = 0
    x = df_x[:, frameInd]
    y = df_y[:, frameInd]
    for i in range(0, nparts):
        # print(x[i])
        # print(y[i])
        topEdge = int(y[i]) - int(roi * 0.5)
        if topEdge < 0:
            topEdge = 0
        bottomEdge = int(y[i]) + int(roi * 0.5)
        if bottomEdge > int(height):
            bottomOvershot = bottomEdge - int(height)
            bottomEdge = int(height)
        leftEdge = int(x[i]) - int(roi * 0.5)
        if leftEdge < 0:
            leftEdge = 0
        rightEdge = int(x[i]) + int(roi * 0.5)
        if rightEdge > int(width):
            rightOvershot = rightEdge - int(width)
            rightEdge = int(width)

        if gpu:
            cfr = cp_frame[topEdge:bottomEdge, leftEdge:rightEdge]
            cfr_r = cfr[..., 0]
            cfr_g = cfr[..., 1]
            cfr_b = cfr[..., 2]
            shapeCfr = cfr_r.shape

            # Correct (adding zeros) to make a square shape in case it is not roixroi due to negative values in above section substractions
            if topEdge == 0:
                rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
                cfr_r = cp.vstack((rw, cfr_r))
                cfr_g = cp.vstack((rw, cfr_g))
                cfr_b = cp.vstack((rw, cfr_b))
                shapeCfr = cfr_r.shape
            if bottomOvershot > 0:
                rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
                cfr_r = cp.vstack((cfr_r, rw))
                cfr_g = cp.vstack((cfr_g, rw))
                cfr_b = cp.vstack((cfr_b, rw))
                shapeCfr = cfr_r.shape
            if leftEdge == 0:
                col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
                cfr_r = cp.hstack((col, cfr_r))
                cfr_g = cp.hstack((col, cfr_g))
                cfr_b = cp.hstack((col, cfr_b))
                shapeCfr = cfr_r.shape
            if rightOvershot > 0:
                col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
                cfr_r = cp.hstack((cfr_r, col))
                cfr_g = cp.hstack((cfr_g, col))
                cfr_b = cp.hstack((cfr_b, col))
                shapeCfr = cfr_r.shape

            cfr_normal[..., 0] = cfr_r
            cfr_normal[..., 1] = cfr_g
            cfr_normal[..., 2] = cfr_b
            ROI_array[..., i] = cfr_normal

        elif not gpu:
            cfr = frame_cropped[topEdge:bottomEdge, leftEdge:rightEdge]
            # cv2.imshow('cfr', cfr)
            shapeCfr = cfr.shape
            # print(shapeCfr)

            # Correct (adding zeros) to make a square shape in case it is not roixroi due to negative values in above section substractions
            if topEdge == 0:
                rw = np.absolute(shapeCfr[0] - roi)
                cfr = cv2.copyMakeBorder(cfr, rw, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                shapeCfr = cfr.shape
            if bottomOvershot > 0:
                rw = np.absolute(shapeCfr[0] - roi)
                cfr = cv2.copyMakeBorder(cfr, 0, rw, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                shapeCfr = cfr.shape
            if leftEdge == 0:
                col = np.absolute(shapeCfr[1] - roi)
                cfr = cv2.copyMakeBorder(cfr, 0, 0, col, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                shapeCfr = cfr.shape
            if rightOvershot > 0:
                col = np.absolute(shapeCfr[1] - roi)
                cfr = cv2.copyMakeBorder(cfr, 0, 0, 0, col, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                shapeCfr = cfr.shape
            ROI_array[..., i] = cfr

    # if gpu is not None:
    #     ROI_array = cp.asnumpy(ROI_array)

    return ROI_array

# HSV分割
def getHSV(img, lower_color, upper_color, kernel):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, lower_color, upper_color)
    # mask_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening, iterations=3)
    # mask_dilating = cv2.dilate(mask_opening, kernel, iterations=2)
    # mask_median = cv2.medianBlur(mask_dilating, 3)
    return mask

kernel_opening = np.ones((3, 3), np.uint8)
def getColor(Img):
    lower_color = np.array([30, 50, 90])
    upper_color = np.array([200, 200, 200])
    mask = cv2.inRange(Img, lower_color, upper_color)
    mask = 255 - mask
    # mask_dilating = cv2.erode(mask, kernel_opening, iterations=2)
    mask_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening, iterations=2)
    return mask_opening

def smooth_2d(M, winSm):
    Msm = savgol_filter(M, window_length=winSm, polyorder=0)
    return Msm

# 获取ROI裁剪中心
def getting_crop_center(frRec, TimeWindow, newSize):

    for i in range(0, TimeWindow):

        frameFloat = frRec[..., i]

        # frameVect = rs.reshape(1, int(newSize[0]) * int(newSize[1]))
        # frameVectFloat = frameVect.astype(float)

        if i == 0:
            previousFrame = frameFloat
            frameDiffComm = previousFrame * 0

        if i > 0:
            frameDiffComm = frameDiffComm + cp.absolute(frameFloat - previousFrame)
            previousFrame = frameFloat

    # Find the index of the first pixel in the frame that shows the highest difference in intensity respect to the previous frame
    indMaxDiff = int(cp.argmax(frameDiffComm))
    rowMaxDiff = int(indMaxDiff / newSize[0])
    colMaxDiff = indMaxDiff % newSize[0]

    # Find the value of the pixel in the frame that shows the highest difference in intensity respect to the previous frame
    maxMovement = cp.max(frameDiffComm)
    # print(maxMovement)

    return maxMovement, rowMaxDiff, colMaxDiff


def subtract_average(frameVectRec, dim, gpu):

    if gpu:
        shFrameVectRec = cp.shape(frameVectRec)  # 获取shape
        averageSubtFrameVecRec = cp.zeros((shFrameVectRec[0], shFrameVectRec[1]))  # 生成一个同shape的全0矩阵

        if dim == 0:
            averageVect = cp.mean(frameVectRec, 0)  # 计算cfr每一列的均值

        if dim == 1:
            averageVect = cp.mean(frameVectRec, 1)

    if not gpu:
        shFrameVectRec = np.shape(frameVectRec)  # 获取shape
        averageSubtFrameVecRec = np.zeros((shFrameVectRec[0], shFrameVectRec[1]))  # 生成一个同shape的全0矩阵

        if dim == 0:
            averageVect = np.mean(frameVectRec, 0)  # 计算cfr每一列的均值

        if dim == 1:
            averageVect = np.mean(frameVectRec, 1)

    if dim == 0:
        for i in range(0, shFrameVectRec[0]):
            averageSubtFrameVecRec[i, :] = frameVectRec[i, :] - averageVect  # 将cfr的每一行都减去均值

    if dim == 1:
        for i in range(0, shFrameVectRec[1]):
            averageSubtFrameVecRec[:, i] = frameVectRec[:, i] - averageVect

    return averageSubtFrameVecRec  # 这里返回的矩阵维数与传入矩阵相同，每一行都减去了平均值

# 根据ROI裁剪中心裁剪RGB ROI
def crop_rgb_frame(frRec, TimeWindow, newSize, roi, rowMaxDiff, colMaxDiff):
    rgb_cfrRec = cp.zeros((roi, roi, 3, TimeWindow))
    # cfr_r = cp.zeros((roi, roi))
    # cfr_g = cp.zeros((roi, roi))
    # cfr_b = cp.zeros((roi, roi))
    cfr_normal = cp.zeros((roi, roi, 3))
    bottomOvershot = 0
    rightOvershot = 0

    topEdge = rowMaxDiff - int(roi * 0.5)
    if topEdge < 0:
        topEdge = 0
    bottomEdge = rowMaxDiff + int(roi * 0.5)
    if bottomEdge > int(newSize[0]):
        bottomOvershot = bottomEdge - int(newSize[0])
        bottomEdge = int(newSize[0])
    leftEdge = colMaxDiff - int(roi * 0.5)
    if leftEdge < 0:
        leftEdge = 0
    rightEdge = colMaxDiff + int(roi * 0.5)
    if rightEdge > int(newSize[0]):
        rightOvershot = rightEdge - int(newSize[0])
        rightEdge = int(newSize[0])

    for i in range(0, TimeWindow):
        rf = frRec[..., i]
        # Select the roixroi square from the frame
        cfr = rf[topEdge:bottomEdge, leftEdge:rightEdge]
        cfr_r = cfr[..., 0]
        cfr_g = cfr[..., 1]
        cfr_b = cfr[..., 2]
        shapeCfr = cfr_r.shape

        # Correct (adding zeros) to make a square shape in case it is not roixroi due to negative values in above section substractions
        if topEdge == 0:
            rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr_r = cp.vstack((rw, cfr_r))
            cfr_g = cp.vstack((rw, cfr_g))
            cfr_b = cp.vstack((rw, cfr_b))
            shapeCfr = cfr_r.shape
        if bottomOvershot > 0:
            rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr_r = cp.vstack((cfr_r, rw))
            cfr_g = cp.vstack((cfr_g, rw))
            cfr_b = cp.vstack((cfr_b, rw))
            shapeCfr = cfr_r.shape
        if leftEdge == 0:
            col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr_r = cp.hstack((col, cfr_r))
            cfr_g = cp.hstack((col, cfr_g))
            cfr_b = cp.hstack((col, cfr_b))
            shapeCfr = cfr_r.shape
        if rightOvershot > 0:
            col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr_r = cp.hstack((cfr_r, col))
            cfr_g = cp.hstack((cfr_g, col))
            cfr_b = cp.hstack((cfr_b, col))
            shapeCfr = cfr_r.shape

        cfr_normal[..., 0] = cfr_r
        cfr_normal[..., 1] = cfr_g
        cfr_normal[..., 2] = cfr_b
        rgb_cfrRec[..., i] = cfr_normal

    rgb_cfrRec = cp.asnumpy(rgb_cfrRec)
    # cv2.imshow('cfr', rgb_cfrRec[..., 1]/255)

    return rgb_cfrRec

# 根据ROI裁剪中心裁剪gray ROI
def crop_gary_frame_cupy(frRec, TimeWindow, newSize, roi, rowMaxDiff, colMaxDiff):
    gray_cfrRec = cp.zeros((roi, roi, TimeWindow)) #预分配ROI三维矩阵空间
    # Calculate a roixroi square around the pixel of maximum intensity difference
    bottomOvershot = 0
    rightOvershot = 0

    topEdge = rowMaxDiff - int(roi * 0.5)
    if topEdge < 0:
        topEdge = 0
    bottomEdge = rowMaxDiff + int(roi * 0.5)
    if bottomEdge > int(newSize[0]):
        bottomOvershot = bottomEdge - int(newSize[0])
        bottomEdge = int(newSize[0])
    leftEdge = colMaxDiff - int(roi * 0.5)
    if leftEdge < 0:
        leftEdge = 0
    rightEdge = colMaxDiff + int(roi * 0.5)
    if rightEdge > int(newSize[0]):
        rightOvershot = rightEdge - int(newSize[0])
        rightEdge = int(newSize[0])

    for i in range(0, TimeWindow):
        rf = frRec[..., i]
        # Select the roixroi square from the frame
        cfr = rf[topEdge:bottomEdge, leftEdge:rightEdge]
        shapeCfr = cfr.shape

        # Correct (adding zeros) to make a square shape in case it is not roixroi due to negative values in above section substractions
        if topEdge == 0:
            rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr = cp.vstack((rw, cfr))
            shapeCfr = cfr.shape
        if bottomOvershot > 0:
            rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr = cp.vstack((cfr, rw))
            shapeCfr = cfr.shape
        if leftEdge == 0:
            col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr = cp.hstack((col, cfr))
            shapeCfr = cfr.shape
        if rightOvershot > 0:
            col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr = cp.hstack((cfr, col))
            shapeCfr = cfr.shape

        gray_cfrRec[..., i] = cfr

    # cv2.imshow('cfr_g', cp.asnumpy(gray_cfrRec[..., 1] / 255))

    return gray_cfrRec

# 获取gray ROI帧窗口
def getting_frame_record_cupy(frRec, TimeWindow, newSize, roi):

    cfrRec = cp.zeros((roi, roi, TimeWindow)) #预分配ROI三维矩阵空间

    for i in range(0, TimeWindow):

        frameFloat = frRec[..., i]

        # frameVect = rs.reshape(1, int(newSize[0]) * int(newSize[1]))
        # frameVectFloat = frameVect.astype(float)

        if i == 0:
            previousFrame = frameFloat
            frameDiffComm = previousFrame * 0

        if i > 0:
            frameDiffComm = frameDiffComm + cp.absolute(frameFloat - previousFrame)
            previousFrame = frameFloat

    # Find the index of the first pixel in the frame that shows the highest difference in intensity respect to the previous frame
    indMaxDiff = int(cp.argmax(frameDiffComm))
    rowMaxDiff = int(indMaxDiff / newSize[0])
    colMaxDiff = indMaxDiff % newSize[0]

    # Find the value of the pixel in the frame that shows the highest difference in intensity respect to the previous frame
    maxMovement = cp.max(frameDiffComm)
    # print(maxMovement)

    # Calculate a roixroi square around the pixel of maximum intensity difference
    bottomOvershot = 0
    rightOvershot = 0

    topEdge = rowMaxDiff - int(roi * 0.5)
    if topEdge < 0:
        topEdge = 0
    bottomEdge = rowMaxDiff + int(roi * 0.5)
    if bottomEdge > int(newSize[0]):
        bottomOvershot = bottomEdge - int(newSize[0])
        bottomEdge = int(newSize[0])
    leftEdge = colMaxDiff - int(roi * 0.5)
    if leftEdge < 0:
        leftEdge = 0
    rightEdge = colMaxDiff + int(roi * 0.5)
    if rightEdge > int(newSize[0]):
        rightOvershot = rightEdge - int(newSize[0])
        rightEdge = int(newSize[0])

    for i in range(0, TimeWindow):

        rf = frRec[..., i]

        # Select the roixroi square from the frame
        cfr = rf[topEdge:bottomEdge, leftEdge:rightEdge]
        shapeCfr = cfr.shape
        # print(shapeCfr)

        # Correct (adding zeros) to make a square shape in case it is not roixroi due to negative values in above section substractions
        if topEdge == 0:
            rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr = cp.vstack((rw, cfr))
            shapeCfr = cfr.shape
        if bottomOvershot > 0:
            rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr = cp.vstack((cfr, rw))
            shapeCfr = cfr.shape
        if leftEdge == 0:
            col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr = cp.hstack((col, cfr))
            shapeCfr = cfr.shape
        if rightOvershot > 0:
            col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr = cp.hstack((cfr, col))
            shapeCfr = cfr.shape

        cfrRec[..., i] = cfr

        # cccc = cp.asnumpy(cfr)
        # cccc /= 255
        # cv2.imshow('cf', cccc)
        # cv2.waitKey(0)

    return maxMovement, cfrRec

# 生成时空特征图像
def create_ST_image_cupy(cfrRec, TimeWindow, roi):
    # Quoted from ABRS
    # Written by Primoz Ravbar
    # Modified by Augusto Escalante
    # Finally modified by ZL Zhang

    diff_cfr = cp.zeros((roi, roi))
    cfrVectRec = cp.zeros((TimeWindow, roi**2))
    imRaw = cfrRec[..., TimeWindow - 1]
    # imRaw = imRaw.astype('uint')
    # print(cfrVectRec.shape)
    for i in range(0, TimeWindow):
        # print(x.shape)
        cfrVectRec[i] = cp.reshape(cfrRec[..., i], (1, roi**2))[0]
        if i < TimeWindow-1:
            diff_cfr += cp.absolute(cfrRec[..., i] - cfrRec[..., i+1])

    diff = diff_cfr / cp.max(cp.max(cp.reshape(diff_cfr, roi**2)))
    diff[diff < 0.2] = 0
    cG = center_of_gravity(cfrVectRec)  # 通过快速傅立叶变换对图片集进行处理

    averageSubtFrameVecRec = subtract_average(cfrVectRec, 0)  # 得到一个减去均值后的图片集数据

    totVar = cp.sum(cp.absolute(averageSubtFrameVecRec), axis=0)  # averageSubtFrameVecRec取绝对值后再求每一列的和
    imVar = cp.reshape(totVar, (roi, roi))
    imVarNorm = imVar / cp.max(cp.max(imVar))
    imVarBin = cp.zeros((roi, roi))
    imVarBin[imVarNorm > 0.10] = 1

    I = (cp.reshape(cG, (roi, roi))-1) * imVarBin
    I = cp.nan_to_num(I)
    I = cp.clip(I, 0, 1)
    IN = cp.asnumpy(I)
    # plt.imshow(I)
    # plt.show()
    sM = I.reshape((roi*roi))
    sM = cp.nan_to_num(sM)

    if np.max(sM) > 0:
        sMNorm = sM/np.max(sM)
    else:
        sMNorm = sM

    sMNorm = cp.asnumpy(sMNorm)
    imSTsm = smooth_2d(sMNorm, 3)
    I_RS = np.reshape(imSTsm, (roi, roi))
    # I_RS[I_RS < 0.5] = 0

    # I_RS = cp.reshape(IN, (roi, roi))
    # # cv2.imshow('I_RS', I_RS)
    # I_RS = cp.asnumpy(I_RS)
    # imSTsm = smooth_2d(I_RS, 3)
    imRaw = cp.asnumpy(imRaw).astype('uint8')*255
    diff = cp.asnumpy(diff)
    imdiff = smooth_2d(diff, 3)

    rgbArray = np.zeros((roi, roi, 3), 'uint8')
    rgbArray[..., 0] = imRaw # blue channel for cv2.imshow()/ red channel for plt.imshow(): Difference with average of windowST frames
    rgbArray[..., 1] = 0# green channel for cv2.imshow() and plt.imshow(): Insects informations
    rgbArray[..., 2] = I_RS * 255 # red channel for cv2.imshow()/ blue channel for plt.imshow(): Motionless parts

    imST = rgbArray
    # cv2.imshow('im', imST)
    return imST

# 生成时空特征图像（取RGB ROI中的0，2两个通道）
def create_ST_image_ch2(cfrRec, rgbRec, TimeWindow, roi, gpu=False):
    # Quoted from ABRS
    # Written by Primoz Ravbar
    # Modified by Augusto Escalante
    # Finally modified by ZL Zhang

    if gpu:
        cfrVectRec = cp.zeros((TimeWindow, roi ** 2))
        imVarBin = cp.zeros((roi, roi))
        for i in range(0, TimeWindow):
            # print(x.shape)
            cfrVectRec[i] = cp.reshape(cfrRec[..., i], roi**2)
    if not gpu:
        cfrVectRec = np.zeros((TimeWindow, roi ** 2))
        imVarBin = np.zeros((roi, roi))
        for i in range(0, TimeWindow):
            # print(x.shape)
            cfrVectRec[i] = np.reshape(cfrRec[..., i], roi**2)

    imRaw = cfrRec[..., TimeWindow - 1]
    channel0 = rgbRec[..., TimeWindow - 1][..., 0]
    # channel2 = rgbRec[..., TimeWindow - 1][..., 2]
    # imRaw = imRaw.astype('uint')
    # print(cfrVectRec.shape)

    cG = center_of_gravity(cfrVectRec, gpu)  # 通过快速傅立叶变换对图片集进行处理

    averageSubtFrameVecRec = subtract_average(cfrVectRec, 0, gpu)  # 得到一个减去均值后的图片集数据

    if gpu:
        totVar = cp.sum(cp.absolute(averageSubtFrameVecRec), axis=0)  # averageSubtFrameVecRec取绝对值后再求每一列的和
        imVar = cp.reshape(totVar, (roi, roi))
        imVarNorm = imVar / cp.max(cp.max(imVar))
        imVarBin[imVarNorm > 0.10] = 1
        I = (cp.reshape(cG, (roi, roi)) - 1) * imVarBin
        I = cp.nan_to_num(I)
        I = cp.clip(I, 0, 1)
        # IN = cp.asnumpy(I)
        # plt.imshow(I)
        # plt.show()
        sM = I.reshape((roi ** 2))
        sM = cp.nan_to_num(sM)

        if np.max(sM) > 0:
            sMNorm = sM / np.max(sM)
        else:
            sMNorm = sM

        sMNorm = cp.asnumpy(sMNorm)
        imSTsm = smooth_2d(sMNorm, 3)
        I_RS = np.reshape(imSTsm, (roi, roi))
        # I_RS[I_RS < 0.5] = 0

        # I_RS = cp.reshape(IN, (roi, roi))
        # # cv2.imshow('I_RS', I_RS)
        # I_RS = cp.asnumpy(I_RS)
        # imSTsm = smooth_2d(I_RS, 3)
        imRaw = cp.asnumpy(imRaw).astype('uint8') * 255
        channel0 = cp.asnumpy(channel0).astype('uint8')

    if not gpu:
        totVar = np.sum(np.absolute(averageSubtFrameVecRec), axis=0)  # averageSubtFrameVecRec取绝对值后再求每一列的和
        imVar = np.reshape(totVar, (roi, roi))
        imVarNorm = imVar / np.max(np.max(imVar))
        imVarBin[imVarNorm > 0.10] = 1
        I = (np.reshape(cG, (roi, roi)) - 1) * imVarBin
        I = np.nan_to_num(I)
        I = np.clip(I, 0, 1)

        sM = I.reshape((roi ** 2))
        sM = np.nan_to_num(sM)

        if np.max(sM) > 0:
            sMNorm = sM / np.max(sM)
        else:
            sMNorm = sM

        imSTsm = smooth_2d(sMNorm, 3)
        I_RS = np.reshape(imSTsm, (roi, roi))


        imRaw = imRaw.astype('uint8') * 255
        channel0 = channel0.astype('uint8')

    rgbArray = np.zeros((roi, roi, 3), 'uint8')
    rgbArray[..., 0] = imRaw # blue channel for cv2.imshow()/ red channel for plt.imshow(): Difference with average of windowST frames
    rgbArray[..., 1] = channel0 # green channel for cv2.imshow() and plt.imshow(): Insects informations
    rgbArray[..., 2] = I_RS * 255 # red channel for cv2.imshow()/ blue channel for plt.imshow(): Motionless parts

    imST = rgbArray
    # cv2.imshow('im', imST)

    return imST

def getting_frame_record(frRec, startWin, endWin, fb, newSize, roi, CVNsize, tarRec):
    # Quoted from ABRS
    # Written by Primoz Ravbar
    # Modified by Augusto Escalante
    # Finally modified by ZL Zhang

    # Subdivide (or not, fb==0) the video frame in order to analyze each arena (for the authors setup that consist on
    # 4 plates each with a fly in the same frame)
    for i in range(startWin, endWin):

        frame = frRec[i, :]
        # Nothing changes from frame to gray, they are identical columns
        # gray = frame.reshape(newSize[0] * newSize[1])
        gray = cp.reshape(frame, (1, newSize[0] * newSize[1]))

        if fb == 0:
            rf = gray
        if fb == 1:
            rf = gray[0:200, 0:200]

        if fb == 2:
            rf = gray[0:200, 200:400]

        if fb == 3:
            rf = gray[200:400, 0:200]

        if fb == 4:
            rf = gray[200:400, 200:400]

        rs = rf

        # frameVect = rs.reshape(1, int(newSize[0]) * int(newSize[1]))
        # frameVectFloat = frameVect.astype(float)
        frameVectFloat = cp.reshape(rs, (1, int(newSize[0]) * int(newSize[1])))

        if i == startWin:
            previousFrame = frameVectFloat
            frameDiffComm = previousFrame * 0
            frameVectFloatRec = frameVectFloat

        if i > startWin:
            frameDiffComm = frameDiffComm + cp.absolute(frameVectFloat - previousFrame)
            frameVectFloatRec = cp.vstack((frameVectFloatRec, frameVectFloat))
            previousFrame = frameVectFloat

    # Find the index of the first pixel in the frame that shows the highest difference in intensity respect to the previous frame
    indMaxDiff = cp.argmax(frameDiffComm)

    rowMaxDiff = cp.floor(indMaxDiff / int(newSize[0]))
    colMaxDiff = indMaxDiff - (rowMaxDiff * int(newSize[0]))

    rowMaxDiff = rowMaxDiff.astype(int)
    colMaxDiff = colMaxDiff.astype(int)

    # Find the value of the pixel in the frame that shows the highest difference in intensity respect to the previous frame
    maxMovement = cp.max(frameDiffComm)

    for i in range(0, (endWin - startWin)):

        # Make frameVectFloatRec square
        # rs = frameVectFloatRec[i, :].reshape(int(newSize[0]), int(newSize[0]))
        rs = cp.reshape(frameVectFloatRec[i, :], (int(newSize[0]), int(newSize[0])))
        # rt = tarRec[i, :].reshape(int(newSize[0]), int(newSize[0]))
        rt = cp.reshape(tarRec[i, :], (int(newSize[0]), int(newSize[0])))
        # Calculate a roixroi square around the pixel of maximum intensity difference
        bottomOvershot = 0
        rightOvershot = 0

        topEdge = rowMaxDiff - int(roi * 0.5)
        if topEdge < 0:
            topEdge = 0
        bottomEdge = rowMaxDiff + int(roi * 0.5)
        if bottomEdge > int(newSize[0]):
            bottomOvershot = bottomEdge - int(newSize[0])
            bottomEdge = int(newSize[0])
        leftEdge = colMaxDiff - int(roi * 0.5)
        if leftEdge < 0:
            leftEdge = 0
        rightEdge = colMaxDiff + int(roi * 0.5)
        if rightEdge > int(newSize[0]):
            rightOvershot = rightEdge - int(newSize[0])
            rightEdge = int(newSize[0])

        # Select the roixroi square from the frame
        cfr = rs[topEdge:bottomEdge, leftEdge:rightEdge]
        shapeCfr = cfr.shape
        tfr = rt[topEdge:bottomEdge, leftEdge:rightEdge]

        # Correct (adding zeros) to make a square shape in case it is not roixroi due to negative values in above section substractions
        if topEdge == 0:
            rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr = cp.vstack((rw, cfr))
            tfr = cp.vstack((rw, tfr))
            shapeCfr = cfr.shape
        if bottomOvershot > 0:
            rw = cp.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr = cp.vstack((cfr, rw))
            tfr = cp.vstack((tfr, rw))
            shapeCfr = cfr.shape
        if leftEdge == 0:
            col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr = cp.hstack((col, cfr))
            tfr = cp.hstack((col, tfr))
            shapeCfr = cfr.shape
        if rightOvershot > 0:
            col = cp.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr = cp.hstack((cfr, col))
            tfr = cp.hstack((tfr, col))
            shapeCfr = cfr.shape

        # # Resize roixroi to CVNsizexCVNsize:
        # smallcfr = cv2.resize(cfr, (CVNsize, CVNsize))
        # smalltfr = cv2.resize(tfr, (CVNsize, CVNsize))
        # cfrVect = smallcfr.reshape(1, CVNsize * CVNsize)
        # tfrVect = smalltfr.reshape(1, CVNsize * CVNsize)
        cfrVect = cp.reshape(cfr, (1, CVNsize * CVNsize))
        tfrVect = cp.reshape(tfr, (1, CVNsize * CVNsize))
        # cv2.destroyAllWindows()

        if i == 0:
            cfrVectRec = cfrVect
            tfrVectRec = tfrVect
        if i > 0:
            cfrVectRec = cp.vstack((cfrVectRec, cfrVect))
            tfrVectRec = cp.vstack((tfrVectRec, tfrVect))

    return maxMovement, cfrVectRec, tfrVectRec, frameVectFloatRec

# 傅立叶变换
def  center_of_gravity(cfrRec, gpu=False):
    # Quoted from ABRS
    # Written by Primoz Ravbar
    # Finally modified by ZL Zhang
    # sh = np.shape(cfrVectRec)
    sh = cfrRec.shape

    if gpu:
        F=cp.absolute(cp.fft.fft(cfrRec, axis=0))  #对cfr进行快速傅立叶变换之后取绝对值

        av = cp.zeros((1, sh[0])) #建一个行向量，长度为windowST的值
        av[0,:] = cp.arange(1, sh[0]+1) #给它赋值，1到ST
        A = cp.repeat(av,sh[1], axis=0)  #把上面那个矩阵行数扩展到size的平方，每一行的值都等于上面赋值的内容

        FA = F*cp.transpose(A)  #F和A的转置相乘（对应位置的元素直接相乘），F与A的转置维数相同
        # print(FA.shape)
        sF = cp.sum(F,axis=0)  #把F每一列的元素加起来
        sFA = cp.sum(FA,axis=0)  #把FA每一列的元素加起来
        # print(sF.shape)
        cG = sFA / sF

    elif not gpu:
        F = np.absolute(np.fft.fft(cfrRec, axis=0))  # 对cfr进行快速傅立叶变换之后取绝对值

        av = np.zeros((1, sh[0]))  # 建一个行向量，长度为windowST的值
        av[0, :] = np.arange(1, sh[0] + 1)  # 给它赋值，1到ST
        A = np.repeat(av, sh[1], axis=0)  # 把上面那个矩阵行数扩展到size的平方，每一行的值都等于上面赋值的内容

        FA = F * np.transpose(A)  # F和A的转置相乘（对应位置的元素直接相乘），F与A的转置维数相同
        # print(FA.shape)
        sF = np.sum(F, axis=0)  # 把F每一列的元素加起来
        sFA = np.sum(FA, axis=0)  # 把FA每一列的元素加起来
        # print(sF.shape)
        np.seterr(divide='ignore', invalid='ignore') #忽略除法警告
        cG = sFA/sF
        # print(cG.shape)

    return cG  #这里返回的是一个列向量

def create_ST_image(cfrVectRec, tfrVectRec, CVNsize):
    # Quoted from ABRS
    # Written by Primoz Ravbar
    # Modified by Augusto Escalante
    # Finally modified by ZL Zhang

    target = tfrVectRec[1, :]
    target = cp.reshape(target, (CVNsize, CVNsize))
    target = target.astype('uint')

    cG = center_of_gravity(cfrVectRec)  # 通过快速傅立叶变换对图片集进行处理

    imRaw = cp.reshape(cfrVectRec[1, :], (CVNsize, CVNsize))
    imRaw = imRaw.astype('uint')

    I = cp.reshape(cG, (CVNsize, CVNsize)) - 1  # 3月10日修改
    I = cp.nan_to_num(I)
    # plt.imshow(I)
    # plt.show()

    IN = cp.clip(I, 0, 1)  # 限制图片矩阵元素值在0，1之间

    I_RS = cp.reshape(IN, (CVNsize, CVNsize))
    # cv2.imshow('I_RS', I_RS)
    I_RS = cp.asnumpy(I_RS)
    target = cp.asnumpy(target)
    imRaw = cp.asnumpy(imRaw)

    rgbArray = np.zeros((CVNsize, CVNsize, 3), 'uint8')
    rgbArray[..., 0] = I_RS * 255  # blue channel for cv2.imshow()/ red channel for plt.imshow(): Difference with average of windowST frames
    rgbArray[..., 1] = target / 1.8  # green channel for cv2.imshow() and plt.imshow(): Insects informations
    rgbArray[..., 2] = imRaw * 255  # red channel for cv2.imshow()/ blue channel for plt.imshow(): Motionless parts

    imST = rgbArray

    return imST

# 随机取帧
def random_create_data(path=r'C:\Users\ZZL\Desktop\桔小实蝇\merge_jxsy.csv'):
    merge_file = pd.read_csv(path)
    video_id = merge_file['视频']
    motion = merge_file['行为']
    start = merge_file['开始帧']
    end = merge_file['结束帧']
    time = merge_file['持续帧数']
    random = pd.DataFrame(columns=['视频', '行为', '帧'])
    for i in range(0, len(motion)):
        if motion[i] == '触角' or motion[i] == '复眼' or motion[i] == '口器':
            label_name = '头部'
        else:
            label_name = motion[i]
        if start[i] == 0:
            frame_list = list(range(int(start[i])+7, int(end[i]) + 1))
        else:
            if end[i] - start[i] == time[i]:
                frame_list = list(range(int(start[i]), int(end[i]) + 1))
        print(label_name)
        # print(i)
        if label_name == '0':
            frame_choice_list = sorted(rd.sample(frame_list, int(time[i]/400)))
        elif label_name == '头部':
            frame_choice_list = sorted(rd.sample(frame_list, int(time[i]/11)))
        elif label_name == '前足':
            frame_choice_list = sorted(rd.sample(frame_list, int(time[i]/10)))
        elif label_name == '前中足':
            frame_choice_list = sorted(rd.sample(frame_list, int(time[i]/12)))
        elif label_name == '翅膀':
            frame_choice_list = sorted(rd.sample(frame_list, int(time[i]/15)))
        elif label_name == '后中足':
            frame_choice_list = sorted(rd.sample(frame_list, int(time[i]/5)))
        elif label_name == '腹部':
            frame_choice_list = sorted(rd.sample(frame_list, int(time[i]/2.5)))
        elif label_name == '后足':
            frame_choice_list = sorted(rd.sample(frame_list, int(time[i] / 5)))
        for f in frame_choice_list:
            random = random.append(pd.DataFrame({'视频': [video_id[i]], '行为': [label_name], '帧': [f]}))
    random.to_csv('random.csv', index=False)
# random_create_data()

def random_delete(path, val_folder, delete_ratio = 0.3): # 随机剪切主文件夹内一定比例的数据到测试集文件内，生成测试集
    file = os.listdir(path)
    delete_list = rd.sample(file, int(len(file)*delete_ratio))
    for file_name in delete_list:
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            shutil.move(file_path, val_folder)
            # print(file_name, ' 已移动')
        else:
            print(file_name, ' 不存在')
# random_delete(r'G:\train\ST\posterior\翅膀', r'G:\val\ST\posterior\翅膀')