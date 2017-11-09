# -*- coding: UTF-8 -*-
from __future__  import division
#import sys
#sys.path.append("/home/xuehao/caffe/python")
import math
import caffe
import numpy as np
import cv2
from collections import Counter
import time,os


#定义sigmod函数
def sigmod(x):
   return 1.0 / (1.0 + math.exp(-x))

#nms算法
def nms(dets, thresh):
    #dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    #thresh:0.3,0.5....
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)#求每个bbox的面积
    order = scores.argsort()[::-1]#对分数进行倒排序
    keep = []#用来保存最后留下来的bbox
    while order.size > 0:
        i = order[0]#无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        #计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        #计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= thresh)[0]
        #因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    return keep

#传图加载检测模型
def load_model(net,test_img):
   input_img = cv2.resize(test_img,(416,416),interpolation=cv2.INTER_AREA)
   ######yolo传图必须是rgb格式且输入矩阵必须为4维1，3，416，416，不得为3，416，416，虽然不报错，但是影响检测效果########
   input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)/255
   input_img = input_img.transpose(2,0,1)
   input_img = input_img.reshape((1,3,416,416))
   out = net.forward_all(data=input_img)
   #模型feature map输出数组############################
   shape = out['conv17'].transpose(0, 3, 2, 1)[0]
   return shape

#处理预测框
def box(shape, test_img):
   box_list = []
   draw_list = []
   for i in range(13):
      for j in range(13):
         anchors_boxs_shape = shape[i][j].reshape((5, cl_num + 5))
         #将每个预测框向量包含信息迭代出来
         for k in range(5):
            anchors_box = anchors_boxs_shape[k]
            #计算实际置信度,阀值处理,anchors_box[7]
            prob = sigmod(anchors_box[4])
            if prob > Confidence:
               #tolist()数组转list
               cls_list = anchors_box[5:cl_num + 5].tolist()
               cls = classes[cls_list.index(max(cls_list))]
               num = cls_list.index(max(cls_list))
               obj_prob = prob
               #yolov2公式推导
               x = (sigmod(anchors_box[0]) + i)/13.0
               y = (sigmod(anchors_box[1]) + j)/13.0
               w = (bias_w[k] * math.exp(anchors_box[2]))/13.0
               h = (bias_h[k] * math.exp(anchors_box[3]))/13.0
               #实际box中心点坐标值
               box_w = w * len(test_img[0])
               box_h = h * len(test_img)
               box_x = x * len(test_img[0])
               box_y = y * len(test_img)
               x1 = int(box_x - box_w * 0.5)
               x2 = int(box_x + box_w * 0.5)
               y1 = int(box_y - box_h * 0.5)
               y2 = int(box_y + box_h * 0.5)
               box_list.append([x1,y1,x2,y2,obj_prob])
               draw_list.append([x1,y1,x2,y2,obj_prob,num])
   if box_list:
      nms_box = nms(np.array(box_list), 0.3)
      draw_box_list = []
      for i in nms_box:
         draw_box_list.append(draw_list[i])
   else:
       draw_box_list = draw_list
   return draw_box_list

#opencv画框
def draw_box(box_list, fps_num,test_img):
   color = [(0, 255, 0), (100, 0, 255), (255,195, 0), (200, 240, 0), (255, 195, 0), (100, 0, 255), (200, 0, 255),
            (255, 0, 0), (57, 172, 0)]
   temp = 0
   for i in box_list:
      if i == '':
          continue
      ####roi接口传图,坐标不得有负数###################
      if i[0] < 0:
         i[0] = 0
      roi_img = test_img[i[1]:(i[1] + (i[3] - i[1])), i[0]:(i[0] + (i[2] - i[0]))]
      if len(roi_img) == 0:
         continue
      temp += 1
      #if 1040 > i[3] > 250 and i[3] and i[2]<1890 and i[0] > 30:
         #if i[5] == 1 or  i[5] == 5:
            #cv2.imwrite("car/2_%s_%s.jpg" %(fps_num,temp), roi_img)
      cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), color[i[5]], 3)
      cv2.putText(test_img, classes[i[5]], (i[0], i[1] - 8), 0, 1, (0, 0, 255), 2)
      print [classes[i[5]], i[4], i[0], i[1], i[2], i[3]]
   #####写图片接口###############
   #cv2.imwrite("%s.jpg" % fps_num, test_img)


#####写视频接口#####################
def viedo():
    cap = cv2.VideoCapture("3.avi")
    # 获得码率及尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 指定写视频的格式, I420-avi, MJPG-mp4
    videoWriter = cv2.VideoWriter('3.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    fps_num = 0
    while True:
        ret, test_img = cap.read()
	#time strat
        start = time.clock() 
        shape = load_model(net, test_img)
        box_list = box(shape, test_img)
        draw_box(box_list, fps_num, test_img)
        end = time.clock()
        print "SPEND_TIME:"+str(end-start)
        print "FPS_NUM:"+str(fps_num)
        fps_num += 1
        #cv2.imshow("capture", test_img)
        videoWriter.write(test_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


#####写图接口#####################
def image(img_filename,fps_num):
    #img_filename = '4.png'
    test_img = cv2.imread(img_filename)
    shape = load_model(net, test_img)
    box_list = box(shape, test_img)
    draw_box(box_list, fps_num, test_img)


if __name__ == "__main__":
    # 置信度设置
   Confidence = 0.35
   classes = ["Color", "Van", "Bus", "Person", "Truck", "Car", "bicycle", "motorbike"]
   cl_num = len(classes)
   # 模型anchors参数，即cell预测框初始 w，h
   bias_w = [0.738768, 2.42204, 4.30971, 10.246, 12.6868]
   bias_h = [0.874946, 2.65704, 7.04493, 4.59428, 11.8741]
   #caffe模型加载#########################
   #检测模型
   caffe.set_mode_gpu()
   net = caffe.Net('yolo-car.prototxt', 'yolo8_v3.caffemodel', caffe.TEST)
   viedo()
   lists = os.listdir('img')
   fps_num = 0
   #for i in lists:
      #image('img/%s'%i,fps_num)
   fps_num += 1

