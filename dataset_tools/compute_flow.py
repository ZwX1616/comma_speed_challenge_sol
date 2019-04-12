# convert the video frames into trainable images

set = 'train'
import cv2
import numpy as np
vc = cv2.VideoCapture('../data/'+set+'.mp4')

count = 1

isread, prev_frame = vc.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = prev_frame[310:310+50,:]
# prev_frame = cv2.resize(prev_frame,(192,64))
# hsv = np.zeros((80,240,3))
# hsv[:,1] = 255
while isread:
	isread, curr_frame = vc.read()
	if isread==False: break
	curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
	curr_frame = curr_frame[310:310+50,:]
	# compute optical flow
	flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	# print(np.mean(ang))
	cv2.imwrite('../data/image_'+set+'/'+str(count)+'.jpg', ang*255/np.max(ang))
	import pdb; pdb.set_trace() ###
	# k = mag.reshape((50*640))
	# s = k[np.argsort(k)]
	# s = np.flip(s/np.max(s))
	# print(s[0:200])
	# import pdb; pdb.set_trace() ###
	prev_frame = curr_frame
	count += 1

print('finish reading '+set+'.mp4')