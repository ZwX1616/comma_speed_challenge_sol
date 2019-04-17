import numpy as np
import cv2

# the area of which we want to obtain the top view 
# (top_left, top_right, bot_left, bot_right)
roi = np.array([[220, 260],
			[220+200, 260],
			[44, 260+80],
			[44+552, 260+80]], dtype=np.float32)

# mapped image dimensions
# simplified
new_height = 194
new_width = 552

# the target area of roi mapping 
target_rect = np.array([[0, 0],
						[new_width - 1, 0],
						[0, new_height - 1],
						[new_width - 1, new_height - 1]], dtype=np.float32)

# compute the perspective transform matrix and apply it
M = cv2.getPerspectiveTransform(roi, target_rect)

# if less than this match, use the mean guess instead
MIN_MATCH_COUNT = 1

cap = cv2.VideoCapture('../data/train.mp4')

f = 0
ret, prev = cap.read()
prev = cv2.warpPerspective(prev, M, (new_width, new_height))
while(ret==True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.warpPerspective(frame, M, (new_width, new_height))
    img_next = frame
    img_prev = prev

    img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
    img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)

    # preprocessing for better contrast
    img_next = cv2.equalizeHist(img_next)
    img_prev = cv2.equalizeHist(img_prev)

    _,img_next_shadow = cv2.threshold(img_next,30,255,cv2.THRESH_BINARY_INV)
    _,img_prev_shadow = cv2.threshold(img_prev,30,255,cv2.THRESH_BINARY_INV)
    _,img_next_highlight = cv2.threshold(img_next,245,255,cv2.THRESH_BINARY)
    _,img_prev_highlight = cv2.threshold(img_prev,245,255,cv2.THRESH_BINARY)
    img_next = img_next_shadow + img_next_highlight
    img_prev = img_prev_shadow + img_prev_highlight
    # img_next = cv2.normalize(img_next, None, 0, 255, cv2.NORM_MINMAX)
    # img_prev = cv2.normalize(img_prev, None, 0, 255, cv2.NORM_MINMAX)

    # Initiate SIFT detector
    # opencv-python contrib
    sift = cv2.xfeatures2d.SIFT_create(350)#(0,3,0.04,16,1.3)

    # find the keypoints and descriptors with SURF
    kp1, des1 = sift.detectAndCompute(img_next,None)
    kp2, des2 = sift.detectAndCompute(img_prev,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)

    v = []
    h = []
    # img_next = cv2.adaptiveThreshold(img_next,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,11,2)

    img_next = cv2.cvtColor(img_next, cv2.COLOR_GRAY2BGR)
    if len(good)>MIN_MATCH_COUNT:
        # (N, 1, 2)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        for i in range(src_pts.shape[0]):
	        cv2.line(img_next, (src_pts[i][0][0], src_pts[i][0][1]), (dst_pts[i][0][0], dst_pts[i][0][1]), (0,0,255), 2)
    else:
        pass

    
    prev = frame
    # Display the resulting frame
    cv2.imshow('Video',img_next)
    f+=1
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


