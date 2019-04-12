import numpy as np
import cv2

# set the displacement search range
# disp of img(t+1)
## shift left or right radius
h_radius = 6
## shift up radius
v_radius = 35

# define when two pixels are "difference"
# feel the magic!
diff_thresh = 16

# [top, left, top+h, left+w] crop of the img_next
# which is used for comparison
compare_area = [v_radius, h_radius, 110-1, 354-h_radius-1]

def approx_disp(img_next, img_prev):
	# img_next: (110, 354, 3)
	# img_prev: (110, 354, 3)
	# output: the delta_v, delta_h that match the patch best
	img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
	img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
	patch = img_next[compare_area[0]:compare_area[2],compare_area[1]:compare_area[3]].astype(np.float32)
	cv2.imwrite('patch.jpg',patch[1])
	diff = np.zeros((1+v_radius,1+2*h_radius)) 
		# axis0 - vertical disp [0,v_r]
		# axis1 - horizontal disp [-h_r,h_r]
	for i in range(1+v_radius):
		for j in range(1+2*h_radius):
			prev_patch = img_prev[compare_area[0]-i:compare_area[2]-i,compare_area[1]-(h_radius-j):compare_area[3]-(h_radius-j)].astype(np.float32)
			# cv2.imwrite('prev_patch.jpg',prev_patch)
			diff[i,j]= np.sum(np.abs(patch-prev_patch)>diff_thresh)

	diff_min = np.min(diff)
	# calculate the most probable disp
	v_min_index, h_min_index = np.where(diff==diff_min)
	delta_v = np.mean(v_min_index) # how much the patch shifted in vertical direction
	delta_h = np.mean(h_min_index)-h_radius # how much the patch shifted in horizontal direction

	return delta_v, delta_h


if __name__ == '__main__':
	# prev = cv2.imread('../data/image_train/19280.jpg')
	# next = cv2.imread('../data/image_train/19281.jpg')
	# v,h = approx_disp(next,prev)
	# print(v)
	# print(h)
	train_end = 19399
	import csv
	wf=open('../data/train_disp_'+str(diff_thresh)+'.txt','w+',newline='') # format: start, end
	writer=csv.writer(wf)
	for i in range(train_end):
		prev = cv2.imread('../data/image_train/'+str(i)+'.jpg')
		next = cv2.imread('../data/image_train/'+str(i+1)+'.jpg')
		v,h = approx_disp(next,prev)
		writer.writerow([str(v),str(h)])
		print('written est disp for frame '+str(i+1), end='\r')
