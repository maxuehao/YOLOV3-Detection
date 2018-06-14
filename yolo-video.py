# -*- coding: UTF-8 -*-
from __future__  import division
import sys
sys.path.append("/data/caffe-segnet-cudnn5/python")
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
	out_list = []
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
		#nms算法
		nms_box = nms(np.array(box_list), 0.3)
      		draw_box_list = []
		for i in nms_box:
			#根据nms筛选后的box_index保留合适的box
         		draw_box_list.append(draw_list[i])
	else:
       		draw_box_list = draw_list
   	
	#为box赋予初始id_label
	global id_num
   	for i in draw_box_list:
       		out_list.append([i[0],i[1],i[2],i[3],i[4],i[5],id_num])
       		id_num += 1
	
	return out_list

def draw_box(box_list, fps_num,test_img):
	color = [(0, 255, 0), (100, 0, 255), (255,195, 0), (200, 240, 0), (255, 195, 0), (100, 0, 255), (200, 0, 255),
		(255, 0, 0), (57, 172, 0)]
	temp = 0
	for i in box_list:
		if i == '':
			continue
		if i[0] < 0:
			i[0] = 0
		roi_img = test_img[i[1]:(i[1] + (i[3] - i[1])), i[0]:(i[0] + (i[2] - i[0]))]
		if len(roi_img) == 0:
         		continue
      		temp += 1
      		#cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), color[i[5]], 3)
      		cv2.putText(test_img, str(i[6]), (i[0], i[1] - 8), 0, 1, (0, 0, 255), 2)
      		#print [classes[i[5]], i[4], i[0], i[1], i[2], i[3]]

#计算两个BOX的IOU重合程度
def calculateIoU(candidateBound, groundTruthBound):
	cx1 = candidateBound[0]
	cy1 = candidateBound[1]
	cx2 = candidateBound[2]
	cy2 = candidateBound[3]
 
	gx1 = groundTruthBound[0]
	gy1 = groundTruthBound[1]
	gx2 = groundTruthBound[2]
	gy2 = groundTruthBound[3]
 
	carea = (cx2 - cx1) * (cy2 - cy1) 
	garea = (gx2 - gx1) * (gy2 - gy1) 
 
	x1 = max(cx1, gx1)
	y1 = max(cy1, gy1)
	x2 = min(cx2, gx2)
	y2 = min(cy2, gy2)
	w = max(0, x2 - x1)
	h = max(0, y2 - y1)
	area = w * h 
	iou = area / (carea + garea - area)
	return iou

def Kalman():
	#kalman
	kalman = cv2.KalmanFilter(8,4)
	#测量矩阵 measurement matrix (H)
	kalman.measurementMatrix =np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]],np.float32)

	#状态传递矩阵 state transition matrix (A) 
	kalman.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],
					    [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)

	#过程噪声协方差矩阵 process noise covariance matrix (Q)
	kalman.processNoiseCov = np.array([ [1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
					    [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)*1e-5
	#先验误差计协方差矩阵 priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)
	kalman.errorCovPre = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
					    [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)*1000	
	#测量噪声协方差矩阵 measurement noise covariance matrix (R)
	kalman.measurementNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)	
	return kalman


#读取视频及摄像头函数
def viedo():
	cap = cv2.VideoCapture("3.avi")
	fps_num = 0
	#创建记录k-1时刻的box缓存数组
	box_buff = []
	while True:
		bbox = []
		ret, test_img = cap.read()
		test_img = cv2.resize(test_img,(1280,720),interpolation=cv2.INTER_AREA)
		shape = load_model(net, test_img)
		box_list = box(shape, test_img)
		start = time.clock()
		print "FPS_NUM:"+str(fps_num)
		#为缓存数组赋予初始值（缓存k-1时刻的BOX信息）
		if box_buff == []:
			box_buff =  box_list
		else:
			#将当前时刻的ｋ的检测框信息依次与K-1时刻的检测框信息做iou计算
			#如果iou大于设定的阈值，则判断为同一物体id
			for i in box_list:
				t = 0
				for j in box_buff:
					iou = calculateIoU(i, j)
					if iou > 0.65:
						t = 1
						ID = j[6]
				if t == 1:
					bbox.append([i[0],i[1],i[2],i[3],i[4],i[5],ID])
					
				else:
		   			bbox.append([i[0],i[1],i[2],i[3],i[4],i[5],i[6]])
			#更新缓存数组
	   		box_buff = bbox
			#初始化kalman滤波器
			for i in bbox:
				x = int(i[0]+0.5*(i[2]-i[0]))  
				y = int(i[1]+0.5*(i[3]-i[1]))
				w = i[2]-i[0]
				h = i[3]-i[1]
				#cv2.circle(test_img,(x, y), 10, (0,255,0), 5) 
				if ('kalman_'+str(i[6]) in vars()):
					current_mes = np.array([[np.float32(x)],[np.float32(y)], [np.float32(w)],[np.float32(h)]])
    					locals()['kalman_'+str(i[6])].correct(current_mes)
    					current_pre = locals()['kalman_'+str(i[6])].predict()
					print current_pre
					cpx, cpy, w, h = current_pre[0],current_pre[1],current_pre[2],current_pre[3]
					#cv2.circle(test_img,(cpx, cpy), 10, (0,0,255), 5)
					cv2.rectangle(test_img, (int(cpx-0.5*w), int(cpy-0.5*h)), (int(cpx+0.5*w), int(cpy+0.5*h)), (183,0,200), 2)  
					
				else: 
    					locals()['kalman_'+str(i[6])] = Kalman()
					current_mes = np.array([[np.float32(x)],[np.float32(y)],[np.float32(w)],[np.float32(h)]])
					locals()['kalman_'+str(i[6])].correct(current_mes)
					current_pre = locals()['kalman_'+str(i[6])].predict()
	
			
	   	end = time.clock()
		#opncv画框
           	draw_box(bbox, fps_num, test_img)
           	print "SPEND_TIME:"+str(end-start)
           	print "=============="
        	fps_num += 1
        	cv2.imshow("capture", test_img)
        	if cv2.waitKey(1) & 0xFF == ord('q'):
            		break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	Confidence = 0.35
	id_num = 0
	classes = ["Color", "Van", "Bus", "Person", "Truck", "Car", "bicycle", "motorbike"]
	cl_num = len(classes)
	bias_w = [0.738768, 2.42204, 4.30971, 10.246, 12.6868]
	bias_h = [0.874946, 2.65704, 7.04493, 4.59428, 11.8741]
	caffe.set_mode_gpu()
	net = caffe.Net('yolo-car.prototxt', 'yolo8_v3.caffemodel', caffe.TEST)
	

	viedo()
