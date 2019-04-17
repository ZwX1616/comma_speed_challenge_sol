import numpy as np
import cv2

thresh_shadow = 35
thresh_highlight = 150
vignette_mask = cv2.imread('../dataset_tools/roi_mask_3.jpg',0)
def outline(img):
    # input image: hist equalized, grayscale
    _,img_shadow = cv2.threshold(img,thresh_shadow,255,cv2.THRESH_BINARY_INV)
    img_shadow = cv2.bitwise_and(img_shadow,vignette_mask)
    _,img_highlight = cv2.threshold(img,thresh_highlight,255,cv2.THRESH_BINARY)
    # img_highlight = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,31,2)
    img_sum = img_shadow + img_highlight
    return img_highlight

    # denoise
    # if (last!=(12,0)):
    #   if abs(delta_v - last[0])>10 or abs(delta_h - last[1])>10: 
    #       delta_v = last[0]
    #       delta_h = last[1]
    #       print('too noisy, use last value instead')
    #       used_last = True

    return delta_v, delta_h, used_last

def filter_match(src_pts,dst_pts):
    out_src = []
    out_dst = []
    for i in range(src_pts.shape[0]):
        disp = np.sqrt((src_pts[i][0][0]-dst_pts[i][0][0])**2+(src_pts[i][0][1]-dst_pts[i][0][1])**2)
        if src_pts[i][0][0]-dst_pts[i][0][0]!=0:
            slope = abs((src_pts[i][0][1]-dst_pts[i][0][1])/(src_pts[i][0][0]-dst_pts[i][0][0]))
        else:
            slope = 50
        if disp>50:
            pass
        else:
            out_src.append(src_pts[i])
            out_dst.append(dst_pts[i])
    if len(out_src)>0:
        return np.stack(out_src,axis=0), np.stack(out_dst,axis=0)
    else: 
        return np.array([]),np.array([])


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


cap = cv2.VideoCapture('../data/test.mp4')

f = 0
ret, prev = cap.read()
prev = cv2.warpPerspective(prev, M, (new_width, new_height))
while(ret==True):
    # Capture frame-by-frame
    ret, frame0 = cap.read()
    frame = cv2.warpPerspective(frame0, M, (new_width, new_height))
    img_next = frame
    img_prev = prev

    # img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
    # img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    img_next = cv2.normalize(img_next, None, 0, 255, cv2.NORM_MINMAX)
    img_prev = cv2.normalize(img_prev, None, 0, 255, cv2.NORM_MINMAX)
    img_next = cv2.bilateralFilter(img_next,9,75,75)
    img_prev = cv2.bilateralFilter(img_prev,9,75,75)

    # preprocessing for better contrast
    # img_next = cv2.equalizeHist(img_next)
    # img_prev = cv2.equalizeHist(img_prev)

    # img_next = outline(img_next)
    # img_prev = outline(img_prev)

    # Initiate SURF detector
    # opencv-python contrib
    surf = cv2.xfeatures2d.SURF_create(50, 4, 3, True, True)

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
        if m.distance < 0.5*n.distance:
            good.append(m)

    v = []
    h = []
    # img_next = cv2.cvtColor(img_next, cv2.COLOR_GRAY2BGR)
    if len(good)>MIN_MATCH_COUNT:
        # (N, 1, 2)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        src_pts, dst_pts = filter_match(src_pts,dst_pts)
        for i in range(src_pts.shape[0]):
	        cv2.line(img_next, (src_pts[i][0][0], src_pts[i][0][1]), (dst_pts[i][0][0], dst_pts[i][0][1]), (0,0,255), 1)
    else:
        pass

    
    prev = frame
    # Display the resulting frame
    f_stack = np.vstack((frame, img_next))
    cv2.imshow('ROI',f_stack)
    cv2.imshow('Video',frame0)
    f+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


