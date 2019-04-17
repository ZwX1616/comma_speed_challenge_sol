import numpy as np
import cv2

def approx_disp_surf(img_next, img_prev, last):
	# img_next: (110, 354, 3)
	# img_prev: (110, 354, 3)
	# output: the most possible delta_v, delta_h

	used_last = False
	# if less than this match, use the mean guess instead
	MIN_MATCH_COUNT = 2

	img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
	img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)

	# preprocessing for better contrast
	img_next = cv2.equalizeHist(img_next)
	img_prev = cv2.equalizeHist(img_prev)

	# Initiate SURF detector
	# opencv-python contrib
	surf = cv2.xfeatures2d.SURF_create(350)

	# find the keypoints and descriptors with SURF
	kp1, des1 = surf.detectAndCompute(img_next,None)
	kp2, des2 = surf.detectAndCompute(img_prev,None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

	v = []
	h = []
	if len(good)>MIN_MATCH_COUNT:
		# (N, 1, 2)
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

		for i in range(src_pts.shape[0]):
			if src_pts[i][0][1] - dst_pts[i][0][1] >= 0 and \
			src_pts[i][0][1] - dst_pts[i][0][1] < 32 and \
			abs(src_pts[i][0][0] - dst_pts[i][0][0]) < 8:
				v.append(src_pts[i][0][1] - dst_pts[i][0][1])
				h.append(abs(src_pts[i][0][0] - dst_pts[i][0][0]))
		if len(v)>0:
			if len(v) < 15:
				delta_v = np.max(np.array(v))
			else:
				delta_v = np.median(np.array(v))
			delta_h = np.median(np.array(h))
		else:
			print('NOT enough usable match, use last value instead')
			delta_v = last[0]
			delta_h = last[1]
			used_last = True

	else:
		print('not enough good match, use last value instead')
		delta_v = last[0]
		delta_h = last[1]
		used_last = True

	# denoise
	# if (last!=(12,0)):
	# 	if abs(delta_v - last[0])>10 or abs(delta_h - last[1])>10: 
	# 		delta_v = last[0]
	# 		delta_h = last[1]
	# 		print('too noisy, use last value instead')
	# 		used_last = True

	return delta_v, delta_h, used_last


if __name__ == '__main__':
	# prev = cv2.imread('../data/image_train/19280.jpg')
	# next = cv2.imread('../data/image_train/19281.jpg')
	# v,h = approx_disp(next,prev)
	# print(v)
	# print(h)
	train_end = 10797# 19399
	import csv
	wf=open('./result/test_disp_surf_3.1.txt','w+',newline='') # format: start, end
	writer=csv.writer(wf)
	last = (12,0)
	for i in range(train_end):
		prev = cv2.imread('../data/image_test/'+str(i)+'.jpg')
		next = cv2.imread('../data/image_test/'+str(i+1)+'.jpg')
		v,h,used_last = approx_disp_surf(next,prev,last)
		writer.writerow([str(v),str(h)])
		if used_last==False:
			last = (v,h)
		else:
			last = (12,0)
		print('written est disp for frame '+str(i+1), end='\r')
