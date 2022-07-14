from imutils import face_utils
import imutils
import dlib
from play_warning import *
from detecting_functions import *
import cv2
import threading
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np  # 数据处理的库 numpy
import argparse
import time
from playsound import playsound
import play_warning
import math


# 眼睛长宽比
# 闪烁阈值
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 6  # 对应眨眼时间大于0.2秒（眨眼时间根据可根据驾驶人年龄进行修改）
# 打哈欠长宽比
# 闪烁阈值
MAR_THRESH = 0.7
MOUTH_AR_CONSEC_FRAMES = 15
# 瞌睡点头
HAR_THRESH = 0.25
NOD_AR_CONSEC_FRAMES = 3
# 初始化帧计数器和非正常眨眼总数
COUNTER = 0
TOTAL = 0
# 初始化帧计数器和打哈欠次数
mCOUNTER = 0
mTOTAL = 0
# 初始化帧计数器和点头总数
hCOUNTER = 0
hTOTAL = 0
# 视频实时帧数
FRAMES = 0
# 警报声音源
warning_path = "C:/Users/mi/Desktop/keras-yolo3-master/Fatigue Driving/warning.wav"
# 是否发出报警
SIGNAL = 0
# 是否允许多线程操作来打开警报
THREAD_OR_NOT = 1
# 线程状态,1表示正在执行
THREAD_STATE = 0

# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

# 初始化Dlib的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
# 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 第三步：分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 第四步：打开cv2 本地摄像头
cap = cv2.VideoCapture(0)
# fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
# print('帧率:{}'.format(fps))

# 从视频流循环帧

while True:
    if FRAMES == 0:
        time0 = time.time()
    # 第五步：进行循环，读取图片，并对图片做维度扩大，并进灰度化
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 第六步：使用detector(gray, 0) 进行脸部位置检测，返回值是矩形的两个坐标
    rects = detector(gray, 0)

    # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息，有多少张脸就循环检测多少次
    for rect in rects:
        # 对第rect张脸进行特征检测，找出关键点
        shape = predictor(gray, rect)

        # 第八步：将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)

        # 第九步：提取左右眼、嘴巴坐标，lStart一类是相应关键点的索引，从第三步得到
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        # 打哈欠嘴部宽度比
        mar = mouth_aspect_ratio(mouth)

        # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
        # 凸包：连成封闭凸多边形的最大外边界（在所有的点中）
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # 第十二步：进行画图操作，用矩形框标注人脸
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        '''
            分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续6次都小于阈值，则表示进行了一次非正常眨眼活动
        '''
        # 第十三步：循环，满足条件的，眨眼次数+1
        if ear < EYE_AR_THRESH:  # 眼睛长宽比：0.2
            COUNTER += 1

        # 如果连续6次都小于阈值，则表示进行了一次非正常眨眼活动
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:  # 阈值：6
                TOTAL += 1
            # 重置眼帧计数器
            COUNTER = 0

        # 第十四步：进行画图操作，68个特征点标识
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # 第十五步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示（脸数，眨眼数，连续眨眼数，眼睛宽度比）
        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "COUNTER: {}".format(COUNTER), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        '''
            计算张嘴宽度比
        '''
        if mar > MAR_THRESH:  # 张嘴阈值0.5
            mCOUNTER += 1
            cv2.putText(frame, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 如果连续mCOUNTER帧都小于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                mTOTAL += 1
            # 重置嘴帧计数器
            mCOUNTER = 0
        cv2.putText(frame, "COUNTER: {}".format(mCOUNTER), (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Yawning: {}".format(mTOTAL), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        '''
            瞌睡点头:获取头部姿态
        '''
        reprojectdst, euler_angle = get_head_pose(shape)

        har = euler_angle[0, 0]  # 取pitch旋转角度
        if har > HAR_THRESH:  # 点头阈值0.25
            hCOUNTER += 1
        else:
            # 如果连续3次都小于阈值，则表示瞌睡点头一次
            if hCOUNTER >= NOD_AR_CONSEC_FRAMES:  # 阈值：3
                hTOTAL += 1
            # 重置点头帧计数器
            hCOUNTER = 0

        # 绘制正方体12轴
        for start, end in line_pairs:
            START = [int(reprojectdst[start][0]), int(reprojectdst[start][1])]
            END = [int(reprojectdst[end][0]), int(reprojectdst[end][1])]
            cv2.line(frame, START, END, (0, 0, 255))
        # 显示角度结果
        cv2.putText(frame, "Pitch: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), thickness=2)  # GREEN
        cv2.putText(frame, "Roll: " + "{:7.2f}".format(euler_angle[1, 0]), (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), thickness=2)  # BLUE
        cv2.putText(frame, "Yaw: " + "{:7.2f}".format(euler_angle[2, 0]), (320, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), thickness=2)  # RED
       # cv2.putText(frame, "Nod: {}".format(hTOTAL), (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # print('嘴巴实时长宽比:{:.2f} '.format(mar) + "\t是否张嘴：" + str([False, True][mar > MAR_THRESH]))
        # print('眼睛实时长宽比:{:.2f} '.format(ear) + "\t是否眨眼：" + str([False, True][COUNTER >= 1]))

    FRAMES = FRAMES + 1  # 根据帧数来交替显示"DON'T SLEEP"
    # 非正常眨眼超过10次或者长时间闭眼判定为疲劳驾驶
    if (TOTAL >= 10 or COUNTER >= 45 or mTOTAL >= 5 or hTOTAL >= 5) and (FRAMES // 10) % 2 == 1:
        cv2.putText(frame, "DON'T SLEEP!!!", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        SIGNAL = 1

    cv2.putText(frame, "Press 'q': Quit", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)

    # 时间过长，变量FRAMES会变得非常大，要及时置零
    if FRAMES == 30:
        time1 = time.time()
        print('30帧运行时间：', (time1-time0))
        FRAMES = 0

    # 窗口显示 show with opencv
    cv2.imshow("Frame", frame)
    # 执行多线程播放警报，只拓展一个线程
    if SIGNAL == 1 and THREAD_OR_NOT == 1:
        music = threading.Thread(target=play, args=(warning_path,))
        music.start()
        THREAD_OR_NOT = 0
        THREAD_STATE = 1
    # print('THREAD_STATE: ', THREAD_STATE) 测试用

    # 按下‘c’键，停止多线程关闭警报
    # 按下‘q’键，关闭窗口
    # 按下‘R’键，信息重置
    key = cv2.waitKey(1)
    if key:
        if key == ord('q' or 'Q'):
            break
        elif key == ord('r' or 'R'):
            TOTAL = 0
        elif key == ord('c' or 'C') and THREAD_STATE == 1:
            stop_thread(music)
            TOTAL = 0
            mTOTAL = 0
            hTOTAL = 0
            THREAD_STATE = 0
            SIGNAL = 0
            THREAD_OR_NOT = 1


# 释放摄像头 release camera
cap.release()
# 销毁所有窗口
cv2.destroyAllWindows()