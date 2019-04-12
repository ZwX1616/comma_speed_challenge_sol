# convert the video frames into trainable images

set = 'train'
import cv2
import numpy as np
vc = cv2.VideoCapture('../data/'+set+'.mp4')

count = 0

isread, prev_frame = vc.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = prev_frame[255:255+100,170:170+300]
prev_frame = cv2.resize(prev_frame,(192,64))
prev_frame = cv2.equalizeHist(prev_frame)

hsv = np.zeros((64,192,3))
hsv[...,1] = 255
while isread:
	isread, curr_frame = vc.read()
	if isread==False: break
	curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
	curr_frame = curr_frame[255:255+100,170:170+300]
	curr_frame = cv2.resize(curr_frame,(192,64))
	curr_frame = cv2.equalizeHist(curr_frame)
	
	# compute optical flow
	# flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.6, 3, 8, 3, 5, 1.1, 0)
	# mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	# hsv[...,0] = ang*180/np.pi/2
	# hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	# # # import pdb; pdb.set_trace() ###
	# bgr = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
	# bgr = cv2.resize(bgr,(192,64))
#	import pdb; pdb.set_trace()
	cv2.imwrite('../data/image_'+set+'/'+str(count)+'.jpg', curr_frame)
	print('written flow for frame '+str(count), end='\r')
	prev_frame = curr_frame
	count += 1

print('finish reading '+set+'.mp4')