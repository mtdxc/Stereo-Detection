import cv2
import numpy as np
import time
import math

import argparse
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
ap.add_argument("-s", "--stereo", type=str, default="sgbm", 
    help="stereo method: sgbm or bm")
ap.add_argument("-c", "--cuda", type=bool, default=True, help="use cuda")
args = vars(ap.parse_args())

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
left_camera_matrix = np.array([[516.5066236,-1.444673028,320.2950423],[0,516.5816117,270.7881873],[0.,0.,1.]])
right_camera_matrix = np.array([[511.8428182,1.295112628,317.310253],[0,513.0748795,269.5885026],[0.,0.,1.]])

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[-0.046645194,0.077595167, 0.012476819,-0.000711358,0]])
right_distortion = np.array([[-0.061588946,0.122384376,0.011081232,-0.000750439,0]])

# 旋转矩阵
R = np.array([[0.999911333,-0.004351508,0.012585312],
              [0.004184066,0.999902792,0.013300386],
              [-0.012641965,-0.013246549,0.999832341]])
# 平移矩阵
T = np.array([-120.3559901,-0.188953775,-0.662073075])

size = (640, 480)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
print(Q)

# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        xyz = threeD[y][x]
        # print("世界坐标是：", xyz[0], xyz[1], xyz[2], "mm")
        print("世界坐标xyz 是：", xyz[0] / 1000.0, xyz[1] / 1000.0, xyz[2] / 1000.0, "m")

        distance = math.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")


# 加载视频文件
capture = cv2.VideoCapture("./car.avi")
WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

# 读取视频
fps = 0.0
ret, frame = capture.read()
useSGBM = args["stereo"] == "sgbm"
useCuda = args["cuda"]
print('use stereo method', args["stereo"], ",cuda=", useCuda)
print('use tracker', args["tracker"])
# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    #"boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    #"tld": cv2.TrackerTLD_create,
    #"medianflow": cv2.TrackerMedianFlow_create,
    #"mosse": cv2.TrackerMOSSE_create
}
# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
# tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
# initialize the bounding box coordinates of the object we are going to track
initBB = None

while ret:
    # 开始计时
    t1 = time.time()
    # 是否读取到了帧，读取到了则为True
    ret, frame = capture.read()
    # 切割为左右两张图片
    width = int(frame.shape[1] / 2)
    height = frame.shape[0]
    frame1 = frame[0:height, 0:width]
    frame2 = frame[0:height, width:width*2]
    
    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    if useSGBM:
        # ------------------------------------SGBM算法----------------------------------------------------------
        #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
        #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
        #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
        #                               取16、32、48、64等
        #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
        #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
        # ------------------------------------------------------------------------------------------------------
        blockSize = 3
        img_channels = 3
        if useCuda:
            stereo = cv2.cuda.createStereoSGM(minDisparity=1,
                                    numDisparities=64,
                                    mode=cv2.STEREO_SGBM_MODE_HH)
        else:
            stereo = cv2.StereoSGBM_create(minDisparity=1,
                                    numDisparities=64,
                                    blockSize=blockSize,
                                    P1=8 * img_channels * blockSize * blockSize,
                                    P2=32 * img_channels * blockSize * blockSize,
                                    disp12MaxDiff=-1,
                                    preFilterCap=1,
                                    uniquenessRatio=10,
                                    speckleWindowSize=100,
                                    speckleRange=100,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM)


    else:
        numberOfDisparities = ((1280 // 8) + 15) & -16  # 640对应是分辨率的宽
        if useCuda:
            stereo = cv2.cuda.createStereoBM(numDisparities=16, blockSize=9)
        else:
            stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)  #立体匹配
        stereo.setROI1(validPixROI1)
        stereo.setROI2(validPixROI2)
        stereo.setPreFilterCap(31)
        stereo.setBlockSize(15)
        stereo.setMinDisparity(0)
        stereo.setNumDisparities(numberOfDisparities)
        stereo.setTextureThreshold(10)
        stereo.setUniquenessRatio(15)
        stereo.setSpeckleWindowSize(100)
        stereo.setSpeckleRange(32)
        stereo.setDisp12MaxDiff(1)

    # 计算视差
    if useCuda:
        img1 = cv2.cuda_GpuMat()
        img2 = cv2.cuda_GpuMat()
        img1.upload(img1_rectified)
        img2.upload(img2_rectified)
        if useSGBM:
            disparity = stereo.compute(img1, img2)
        else:
            stream = cv2.cuda.Stream()
            disparity = stereo.compute(img1, img2, stream)
        disparity = disparity.download()
    else:
        disparity = stereo.compute(img1_rectified, img2_rectified)
        
    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16

    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame1)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
            middle_x = x + int(w / 2)
            middle_y = y + int(h / 2)
            try:
                xyz = threeD[middle_x][middle_y]
                # print("世界坐标是：", xyz[0], xyz[1], xyz[2], "mm")
                str = "(%.2f,%.2f,%.2f)" % (xyz[0] / 1000.0, xyz[1] / 1000.0, xyz[2] / 1000.0)
                cv2.putText(frame, str, (x,y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except:
                print(middle_x, middle_y, "out of range")

    # 鼠标回调事件
    cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, threeD)
    #完成计时，计算帧率
    fps = (fps + (1. / (time.time() - t1))) / 2
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("left", frame1)
    cv2.imshow(WIN_NAME, disp)  # 显示深度图的双目画面

    # 若键盘按下q则退出播放
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("left", frame1, False, False)
        print("selectROI", initBB)
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
        tracker.init(frame1, initBB)
    
# 释放资源
capture.release()

# 关闭所有窗口
cv2.destroyAllWindows()