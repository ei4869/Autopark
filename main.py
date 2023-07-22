import cv2
import numpy as np
from multiprocessing import Process, Queue, Event, Manager
import serial
import time
from PID import *
from threading import Timer
import os,sys
import subprocess
import threading
WIDTH = 320
HEIGHT = 240

Kp1 = 1
Ki1 = 0.01
Kd1 = 0

Kp2 = 1
Ki2 = 0.01
Kd2 = 0.0

def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)

def calculate_parallel_lines_distance(line1, line2):
    """Calculate the distance between two parallel lines"""
    # Lines are given as [x1, y1, x2, y2]
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate coefficients a, b, c1 for the first line
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1*x1 + b1*y1

    # Calculate coefficients a, b, c2 for the second line
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2*x3 + b2*y3

    # Check if lines are parallel
    if a1*b2 != a2*b1:
        raise ValueError('Lines are not parallel')

    # Calculate distance
    distance = abs(c2 - c1) / np.sqrt(a1**2 + b1**2)
    return distance
def calculate_angle(point1, point2):
    """Calculate the angle between two points (in degrees)"""
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

roi1 = [1, 138, 39, 59]
rroi1 = [1, 138, 209,249]
roi3 =  [0, 16, 185, 225]
roi4 = [25, 42, 210, 257]

def img_preprocess(img):
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.GaussianBlur(grayimg, (3, 3), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    can = cv2.Canny(grayimg,100,200)
    
    lower_black = np.array([0, 0, 116])
    upper_black = np.array([255, 255, 255])
    # 根据阈值范围创建掩模 
    mask = cv2.inRange(hsv, lower_black, upper_black)
    # 使用掩模来提取黑色部分
    threimg = cv2.bitwise_and(255*np.ones_like(mask), 255*np.ones_like(mask), mask=mask)

    img1 = grayimg[roi1[0]:roi1[1], roi1[2]:roi1[3]]
    edges1 = cv2.Canny(img1, 100, 200)
    rimg1 = grayimg[rroi1[0]:rroi1[1], rroi1[2]:rroi1[3]]
    redges1 = cv2.Canny(rimg1, 100, 200)
    img3 = grayimg[roi3[0]:roi3[1], roi3[2]:roi3[3]]
    edges3 = cv2.Canny(img3, 100, 200)
    img4 = grayimg[roi4[0]:roi4[1], roi4[2]:roi4[3]]
    edges4 = cv2.Canny(img4, 100, 200)
    
    # 计算每一行的像素值之和
    row_sums1 = np.sum(edges1, axis=1)
    rrow_sums1 = np.sum(redges1, axis=1)

    #计算每一列的像素值之和
    col_sums3 = np.sum(edges3, axis=0)
    col_sums4 = np.sum(edges4, axis=0)

    # 获取像素和排序后的索引
    sorted_rows1 = np.argsort(row_sums1)
    rsorted_rows1 = np.argsort(rrow_sums1)
    sorted_cols3 = np.argsort(col_sums3)
    sorted_cols4 = np.argsort(col_sums4)

    # 找到像素和最大的行和第二大的行
    max_row1 = sorted_rows1[-1]
    second_max_row1 = sorted_rows1[-2]
    rmax_row1 = rsorted_rows1[-1]
    rsecond_rmax_row1 = rsorted_rows1[-2]
    max_col3 = sorted_cols3[-1]
    second_max_col3 = sorted_cols3[-2]
    max_col4 = sorted_cols4[-1]
    second_max_col4 = sorted_cols4[-2]

    a = min(max_row1,second_max_row1)
    second_max_row1 = max(max_row1,second_max_row1)
    max_row1 = a 
    line1_1 = [0+roi1[2],max_row1+roi1[0],img1.shape[1]+roi1[2],max_row1+roi1[0]]
    cv2.line(img,(line1_1[0],line1_1[1]),(line1_1[2],line1_1[3]),(0,255,0),1)
    line1_2 = [0+roi1[2],second_max_row1+roi1[0],img1.shape[1]+roi1[2],second_max_row1+roi1[0]]
    cv2.line(img,(line1_2[0],line1_2[1]),(line1_2[2],line1_2[3]),(0,255,0),1)
    center1 = (img1.shape[1]//2+roi1[2], max_row1+(second_max_row1-max_row1)//2+roi1[0])
    cv2.circle(img, center1, 2, (0, 0, 255), -1)  
    cv2.rectangle(img,(roi1[2],roi1[0]),(roi1[3],roi1[1]),(0,255,0),1)
    cv2.putText(img, 'roi1', (roi1[2],roi1[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

    a = min(rmax_row1,rsecond_rmax_row1)
    rsecond_rmax_row1 = max(rmax_row1,rsecond_rmax_row1)
    rmax_row1 = a
    rline1_1 = [0+rroi1[2],rmax_row1+rroi1[0],rimg1.shape[1]+rroi1[2],rmax_row1+rroi1[0]]
    cv2.line(img,(rline1_1[0],rline1_1[1]),(rline1_1[2],rline1_1[3]),(255,0,0),1)
    rline1_2 = [0+rroi1[2],rsecond_rmax_row1+rroi1[0],rimg1.shape[1]+rroi1[2],rsecond_rmax_row1+rroi1[0]]
    cv2.line(img,(rline1_2[0],rline1_2[1]),(rline1_2[2],rline1_2[3]),(255,0,0),1)
    rcenter1 = (rimg1.shape[1]//2+rroi1[2], rmax_row1+(rsecond_rmax_row1-rmax_row1)//2+rroi1[0])
    cv2.circle(img, rcenter1, 2, (0, 0, 255), -1)
    cv2.rectangle(img,(rroi1[2],rroi1[0]),(rroi1[3],rroi1[1]),(0,255,0),1)
    cv2.putText(img, 'rroi1', (rroi1[2],rroi1[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

    a = min(max_col3,second_max_col3)
    second_max_col3 = max(max_col3,second_max_col3)
    max_col3 = a
    line3_1 = [max_col3+roi3[2],0+roi3[0],max_col3+roi3[2],img3.shape[0]+roi3[0]]
    cv2.line(img,(line3_1[0],line3_1[1]),(line3_1[2],line3_1[3]),(0,255,0),1)
    line3_2 = [second_max_col3+roi3[2],0+roi3[0],second_max_col3+roi3[2],img3.shape[0]+roi3[0]]
    cv2.line(img,(line3_2[0],line3_2[1]),(line3_2[2],line3_2[3]),(0,255,0),1)
    center3 = (max_col3+(second_max_col3-max_col3)//2+roi3[2], img3.shape[0]//2+roi3[0])
    cv2.circle(img, center3, 2, (0, 0, 255), -1)
    cv2.rectangle(img,(roi3[2],roi3[0]),(roi3[3],roi3[1]),(0,255,0),1)
    cv2.putText(img, 'roi3', (roi3[2],roi3[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

    a = min(max_col4,second_max_col4)
    second_max_col4 = max(max_col4,second_max_col4)
    max_col4 = a
    line4_1 = [max_col4+roi4[2],0+roi4[0],max_col4+roi4[2],img4.shape[0]+roi4[0]]
    cv2.line(img,(line4_1[0],line4_1[1]),(line4_1[2],line4_1[3]),(0,255,0),1)
    line4_2 = [second_max_col4+roi4[2],0+roi4[0],second_max_col4+roi4[2],img4.shape[0]+roi4[0]]
    cv2.line(img,(line4_2[0],line4_2[1]),(line4_2[2],line4_2[3]),(0,255,0),1)
    center4 = (max_col4+(second_max_col4-max_col4)//2+roi4[2], img4.shape[0]//2+roi4[0])
    cv2.circle(img, center4, 2, (0, 0, 255), -1)
    cv2.rectangle(img,(roi4[2],roi4[0]),(roi4[3],roi4[1]),(0,255,0),1)
    cv2.putText(img, 'roi4', (roi4[2],roi4[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

    dist1 = calculate_parallel_lines_distance(line3_1,line3_2)
    dist2 = calculate_parallel_lines_distance(line4_1,line4_2)
    # print("检测倒车dist1:",dist1)
    # print("检测侧方停车dist2:",dist2)    
    # print("距离l:",center1[1])
    # print("距离r:",rcenter1[1])
    # print("\n")
    
    cv2.imshow("img",img)
    cv2.imshow("can",can)
    
    return center1[1],rcenter1[1],dist1,dist2
    
QUEUE_MAX_SIZE = 10  # 设置队列的最大长度
def process1(q1, q2,isend):
    # 图像预处理进程
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    cnt = 0
    t1 = time.time()
    while True:
        cnt += 1
        ret, frame = capture.read()
        frame = cv2.flip(frame, -1)
        preprocessed_frame = img_preprocess(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if q1.empty():
            q1.put(frame)
        if q2.empty():
            q2.put(preprocessed_frame)
        if isend.value:
            break
    cv2.destroyAllWindows()

def process2( q2, q3, isend,e2,isflag0,isflag1,isflag2,target):
    global Kp1,Ki1,Kd1,Kp2,Ki2,Kd2
    # PID寻线进程
    pid = CascadePIDController(Kp1,Ki1,Kd1,Kp2,Ki2,Kd2)

    countblack = 0
    tempfg = False
    f1 = True
    f2 = True
    q3.put(('sys','ok'))
    while True:
        if e2.is_set():
            preprocessed_frame = q2.get()
            if isflag0.value:
                cl,cr = preprocessed_frame[0],preprocessed_frame[1]
                output = pid.compute(cl,target.value,cr,target.value)
                q3.put(('pid',output))
            if isflag1.value:
                dist1 = preprocessed_frame[2]
                if dist1>10 and dist1<15 and f1:
                    countblack += 1
                    f1 = False
                if not f1 and dist1<2:
                    f1 = True
                if countblack == 4:
                    time.sleep(0.32)
                    q3.put(('Task','01'))
                    isflag1.value = False
                    countblack = 0
                    isflag0.value = False   #关闭巡线
            if isflag2.value:
                dist2 = preprocessed_frame[3]
                if dist2>10 and dist2<15 and f2:
                    countblack += 1
                    f2 = False
                if not f2 and (dist2<2 or dist2>34):
                    f2 = True
                if countblack == 2:
                    q3.put(('Task','02'))
                    isflag2.value = False
                    countblack = 0
                    isflag0.value = False       
            print("黑线数量countblack:",countblack)
            #打满转过后，恢复直行后，接收到信息后，启动巡线
        if isend.value:
            break

def process3(q3,isend):
    # 串口通信进程
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    while True:

        result_type, result = q3.get()
        if result_type == 'pid':
            if result>=0:
                data = "{:02d}".format(int(result))
                ser.write(("rr"+data+"\r\n").encode('utf-8'))
            else:
                result = -result
                data = "{:02d}".format(int(result))
                ser.write(("ll"+data+"\r\n").encode('utf-8'))
        elif result_type == 'Task':
            time.sleep(0.01)
            ser.write((result+"\r\n").encode('utf-8'))
            print("send ok")
        elif result_type == 'sys':
            ser.write((result+"\r\n").encode('utf-8'))
            print("sys ok")
        if isend.value:
            ser.close()
            break
    #time.sleep(0.05)

def setvalue(isflag2):
    isflag2.value = True
    print("检测侧方停车启动")

def process4(isend,isflag0,isflag2,target):
    #串口接收进程
    ser = serial.Serial("/dev/ttyAMA0", 115200,timeout = 1)
    tempfg2 = True
    while True:
        recv = ser.readline().decode().strip()
        print("收到的数据：", recv)
        if recv == "pid":
            isflag0.value = True
            target.value = 112
            if tempfg2:
                print("yes on pid")
                t1 = Timer(0.18,setvalue,args=(isflag2,))
                t1.start()
                tempfg2 = False
        elif recv == "reboot":
            isend.value = True
            ser.close()
            time.sleep(0.5)
            restart_program()
        ser.flushInput()
        if isend.value:
            ser.close()
            break
        time.sleep(0.1)

if __name__ == '__main__':
    print("程序开始运行")
    q1 = Queue(maxsize=QUEUE_MAX_SIZE)      #存储原始图像
    q2 = Queue(maxsize=QUEUE_MAX_SIZE)      #存储预处理的图像
    q3 = Queue(maxsize=QUEUE_MAX_SIZE)      #存储串口通信的数据

    e2 = Event()
    e2.set()  #开始时开启事件

    manager = Manager()
    isend = manager.Value('b',False)
    isflag0 = manager.Value('b',True)
    isflag1 = manager.Value('b',True)
    isflag2 = manager.Value('b',False)

    target = manager.Value('i',32)

    p1 = Process(target=process1, args=(q1, q2,isend,))
    p2 = Process(target=process2, args=(q2, q3, isend,e2,isflag0,isflag1,isflag2,target,))  
    p3 = Process(target=process3, args=(q3,isend,))
    p4 = Process(target=process4, args=(isend,isflag0,isflag2,target,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

