import numpy as np
import cv2

# the area of which we want to obtain the top view 
# (top_left, top_right, bot_left, bot_right)
# roi = np.array([[200, 280],
# 			[200+240, 280],
# 			[55, 280+75],
# 			[55+530, 280+75]], dtype=np.float32)
roi = np.array([[220, 260],
			[220+200, 260],
			[44, 260+80],
			[44+552, 260+80]], dtype=np.float32)

# mapped image dimensions
# simplified
# new_height = 164
# new_width = 530
new_height = 194
new_width = 552

# the target area of roi mapping 
target_rect = np.array([[0, 0],
						[new_width - 1, 0],
						[0, new_height - 1],
						[new_width - 1, new_height - 1]], dtype=np.float32)

# compute the perspective transform matrix and apply it
M = cv2.getPerspectiveTransform(roi, target_rect)

# map all frames from video
set = 'test'
count = 0
vc = cv2.VideoCapture('../data/'+set+'.mp4')
isread, frame = vc.read()
while isread:
	warped = cv2.warpPerspective(frame, M, (new_width, new_height))
	# warped = cv2.resize(warped,(354,110))
	cv2.imwrite('../data/image_'+set+'/'+str(count)+'.jpg', warped)
	
	print('written flow for frame '+str(count), end='\r')
	count += 1
	isread, frame = vc.read()

print('finish reading '+set+'.mp4')