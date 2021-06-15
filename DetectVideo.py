import os
import pickle
import sys
import time
from pathlib import Path

import cupy as cp
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

import detection_modules
from Ghost import getGhost
from STfeature_modules import create_ST_image_ch2
from STfeature_modules import load_keypoints_info, select_keypoints_ROI, frame_crop

labelname_dict = {0: 0, 1: 'Head', 2: 'ForeLeg', 3: 'ForeMid', 4: 'HindMid',
                  5: 'HindLeg', 6: 'Abdomen', 7: 'Wing'}

# 根据deeplabcut检测的关键点数据和生成的标签提取信息文件，提取RGB和ST roi

scale = 0.6  # 视频缩放比率
roi = 350
TimeWindow = 8
gpu = True  # True or False
t = 12

if roi * scale % 2 != 0:
    print('roi的值需要重新选取，推荐的值为：')
    for roi_i in range(-21, 21):
        for scale_i in range(-5, 5):
            c = (roi + roi_i) * (scale + scale_i / 100) / 2
            c = str(c).split('.')
            if c[1] == '0':
                print('roi=%d,scale=%.2f,c=%d' % (roi + roi_i, scale + scale_i / 100, int(c[0])))
    sys.exit()

roi = int(roi * scale)

file_folder = r'G:\Detect\video'  # 输入单个视频或者视频文件夹
keypoints_base_folder = r'G:\Detect'
save_path = r'F:\51'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
num_class = 8
input = Input((224, 224, 3))
model = getGhost(inputs=input, num_class=num_class)
SGD = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=False)
model.compile(optimizer=SGD,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.load_weights('cnn/3.31.h5')
model.load_weights("D:\\code\\BehaviorDetection\\cnn\\3.31.h5")

video_file_list = []
if os.path.isdir(file_folder):
    video_file = os.listdir(file_folder)
    [video_file_list.append(os.path.join(file_folder, file)) for file in video_file]
elif os.path.isfile(file_folder):
    video_file_list.append(file_folder)

for video in video_file_list:
    video_name = str(Path(video).stem)
    cap = cv2.VideoCapture(video)
    maxframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bodyparts, df_x, df_y, df_likelihood, cropping, cropping_parameters = load_keypoints_info(keypoints_base_folder,
                                                                                              video)
    df_x = df_x * scale
    df_y = df_y * scale
    cropping_parameters_s = []
    for p in cropping_parameters:
        cropping_parameters_s.append(int(p * scale))

    [x1, x2, y1, y2] = cropping_parameters_s
    width = int(x2 - x1)
    height = int(y2 - y1)
    # print(cropping_parameters_s)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30, (width, height))

    startFrame = 0
    endFrame = maxframe

    im3C_arr = np.empty((roi, roi, 3, len(bodyparts)))
    b0_hot = np.empty((num_class, maxframe))
    b0_pre = np.empty((1, maxframe))
    b1_hot = np.empty((num_class, maxframe))
    b1_pre = np.empty((1, maxframe))
    pre = []
    # Preallocation: 使用cupy时所有数据都需要迁移到gpu上
    if gpu:
        frRec = cp.empty((int(height), int(width), 3, TimeWindow))
        cfrRec = cp.empty((roi, roi, TimeWindow, len(bodyparts)))
        rgbRec = cp.empty((roi, roi, 3, len(bodyparts), TimeWindow))
        ROI_arr_gray = cp.empty((roi, roi, len(bodyparts)))
    elif not gpu:
        frRec = np.empty((int(height), int(width), 3, TimeWindow))
        cfrRec = np.empty((roi, roi, TimeWindow, len(bodyparts)))
        rgbRec = np.empty((roi, roi, 3, len(bodyparts), TimeWindow))
        ROI_arr_gray = np.empty((roi, roi, len(bodyparts)))

    for frameInd in range(startFrame, endFrame, 1):

        begin_time = time.clock()

        cap.set(1, frameInd)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(video_width * scale), int(video_height * scale)))

        if gpu:
            currentFrame = cp.array(frame).astype(float)
        elif not gpu:
            currentFrame = frame.astype(float)
        currentFrame_cropped = frame_crop(currentFrame, cropping, cropping_parameters_s)
        # print(currentFrame_cropped.shape)

        for i in range(0, TimeWindow):
            if i == TimeWindow - 1:
                frRec[..., i] = currentFrame_cropped
            else:
                frRec[..., i] = frRec[..., i + 1]

        if frameInd < maxframe and frameInd < df_x.shape[1]:
            j = 0

            for fid in range(0, TimeWindow, 1):

                # cv2.imshow('frame', cv2.resize(frame, (960, 540)))
                # print(frameInd)
                ROI_arr = select_keypoints_ROI(bodyparts, df_x, df_y, frameInd, frRec[..., fid], roi,
                                               gpu)  # 返回的ROI_arr的数据类型由布尔量gpu决定
                # ROI_arr shape: (roi, roi, 3, len(bodyparts))
                rgbRec[..., j] = ROI_arr
                # rgbRec shape: (roi, roi, 3, len(bodyparts), TimeWindow)
                if gpu:
                    ROI_arr_norm = cp.asnumpy(ROI_arr)
                elif not gpu:
                    ROI_arr_norm = ROI_arr
                # print(ROI_arr_gray.shape)
                # Check frames and convert to grayscale:
                if np.size(np.shape(ROI_arr_norm)) >= 2:
                    for nb in range(0, len(bodyparts)):
                        gray = ROI_arr_norm[..., nb].astype(np.float32)
                        if gpu:
                            ROI_arr_gray[..., nb] = cp.array(cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)).astype(float)
                        elif not gpu:
                            ROI_arr_gray[..., nb] = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                else:
                    print('Corrupt frame with less than 2 dimensions!!')
                    if gpu:
                        ROI_arr_gray = np.zeros((roi, roi, len(bodyparts)))  # Fill the corrupt frame with black
                    elif not gpu:
                        ROI_arr_gray = cp.zeros((roi, roi, len(bodyparts)))

                for nb in range(0, len(bodyparts)):
                    cfrRec[..., j, nb] = ROI_arr_gray[..., nb]
                j += 1

                if fid == TimeWindow - 1:
                    rgbRecswap = rgbRec.swapaxes(3, 4)
                    for nb in range(0, len(bodyparts)):
                        cfrRec_clip = cfrRec[..., nb]
                        rgbRec_clip = rgbRecswap[..., nb]
                        # cv2.imshow('t', tfrRec_clip[..., 5])
                        # cv2.waitKey(0)
                        # for k in range(0, TimeWindow):
                        #     cv2.imshow('k', cp.asnumpy(cfrRec_clip[..., k]))
                        #     cv2.waitKey(0)
                        im3C = create_ST_image_ch2(cfrRec_clip.astype('float32'), rgbRec_clip.astype('float32'),
                                                   TimeWindow, roi, gpu)
                        im3C_arr[..., nb] = im3C

            # print(im3C_arr.shape)
            # topEdge, bottomEdge, leftEdge, rightEdge
            dframe = cp.asnumpy(frRec[..., TimeWindow - 1] / 255)
            x = df_x[:, frameInd]
            y = df_y[:, frameInd]
            # print(dframe.shape)
            for b in range(0, len(bodyparts)):
                # print(x[b])
                # print(y[b])
                ex_img = np.expand_dims(cv2.resize(im3C_arr[..., b] / 255, (224, 224)), axis=0)
                one_hot = model.predict(ex_img)
                K.clear_session()
                predict = np.argmax(one_hot)
                if b == 0:
                    b0_hot[..., frameInd] = one_hot
                    b0_pre[..., frameInd] = predict
                if b == 1:
                    b1_hot[..., frameInd] = one_hot
                    b1_pre[..., frameInd] = predict

                    b0_pro = b0_hot[..., frameInd][int(b0_pre[..., frameInd][0])]
                    if b0_pro < 0.4:
                        b0_m = 0
                    elif b0_pro >= 0.4:
                        b0_m = int(b0_pre[..., frameInd])
                    b1_pro = b1_hot[..., frameInd][int(b1_pre[..., frameInd][0])]
                    if b1_pro < 0.4:
                        b1_m = 0
                    elif b0_pro >= 0.4:
                        b1_m = int(b1_pre[..., frameInd])

                    if b0_m == 0 and b1_m == 0:
                        display_pre = 0
                        db = 0
                    elif b0_m == 0:
                        display_pre = b1_m
                        db = 1
                    elif b1_m == 0:
                        display_pre = b0_m
                        db = 0
                    elif b0_m != 0 and b1_m != 0:
                        if b0_pro > b1_pro:
                            display_pre = b0_m
                            db = 0
                        elif b0_pro < b1_pro:
                            display_pre = b1_m
                            db = 1
                    # print(str(labelname_dict[int(display_pre)]), b0_pro, b1_pro)
                    if db == 0:
                        pro = b0_pro
                    elif db == 1:
                        pro = b1_pro
                    # print(rgbRec.shape)
                    cv2.imshow('rgb', cp.asnumpy(rgbRec[..., db, 7]) / 255)
                    # cv2.imshow('rgb2', cp.asnumpy(rgbRec[..., 1, 7]) / 255)
                    cv2.imshow('3c', im3C_arr[..., db] / 255)
                    if int(display_pre) != 0:
                        if display_pre == 1:
                            display_color = (1, 0, 0)
                        elif display_pre == 2:
                            display_color = (1, 0.5, 0)
                        elif display_pre == 3:
                            display_color = (1, 1, 0)
                        elif display_pre == 4:
                            display_color = (0, 1, 0)
                        elif display_pre == 5:
                            display_color = (0, 1, 1)
                        elif display_pre == 6:
                            display_color = (0, 0, 255)
                        elif display_pre == 7:
                            display_color = (0.5, 0, 255)
                        cv2.rectangle(dframe, (int(x[db] - 120 * scale), int(y[db] - 120 * scale)),
                                      (int(x[db] + 120 * scale), int(y[db] + 120 * scale)), display_color, 4)
                        cv2.putText(dframe, str(labelname_dict[int(display_pre)] + '  ' + str(int(pro*100)) + '%'),
                                    (int(x[db] - 60 * scale), int(y[db] + 60 * scale)),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, display_color, 2)

                    pre.append(display_pre)

            # 保存检测视频
            out.write((dframe*255).astype('int8'))
            #################
            cv2.imshow('frame', dframe)

            # end_time = time.clock()  # 记录结束时间
            # run_time = end_time - begin_time
            # print(run_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    flag = 0
    start = []
    end = []
    motion = []

    for i in range(0, len(pre)):
        if i == 0:
            now = pre[i]
            start.append(i)
            motion.append(pre[i])
        if i == len(pre) - 1:
            end.append(i)
        elif now == pre[i]:
            flag = 0
        elif now != pre[i]:
            if flag < t:
                flag += 1
            elif flag >= t:
                flag = 0
                now = pre[i]
                motion.append(pre[i])
                end.append(i - t)
                start.append(i - t)
                if int(now) in [1, 2]:
                    t = 12
                elif int(now) in [0, 5, 6]:
                    t = 20
                elif int(now) in [3, 4, 7]:
                    t = 20
    detection = pd.DataFrame(
        columns=['s分', 's秒', 's帧', 'e分', 'e秒', 'e帧', '持续时间', '持续帧数', '行为', '开始帧', '结束帧'])  # 保存统计数据
    detection = detection_modules.generate_csv(detection, start, end, motion, 25)

    detection.to_csv(os.path.join(save_path, video_name + '.csv'), index=False)

    # print(start)
    # print(end)
    # print(motion)
    print('共有' + str(len(motion)) + '个行为')

    with open(os.path.join(save_path, video_name + '.h5'), 'wb') as f:
        pickle.dump(pre, f)

    # Close all opencv stuff
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()
