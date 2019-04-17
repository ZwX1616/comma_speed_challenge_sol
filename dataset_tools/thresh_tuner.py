# the goal is to outline the shadow and highlight parts of the image
import cv2
def outline(img):
	# input image: hist equalized, grayscale
	thresh_shadow = 25
	thresh_highlight = 250
	_,img_shadow = cv2.threshold(img,thresh_shadow,255,cv2.THRESH_BINARY_INV)
	img_shadow = cv2.bitwise_and(img_shadow,vignette_mask)
	_,img_highlight = cv2.threshold(img,thresh_highlight,255,cv2.THRESH_BINARY)
	
	
	img_sum = img_shadow + img_highlight
	return img_sum



if __name__ == '__main__':
	import cv2
	import numpy as np
	thresh_shadow = 25
	thresh_highlight = 250
	vignette_mask = cv2.imread('roi_mask_3.jpg',0)
	img = cv2.imread('../data/image_train/338.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)
	_,img_shadow = cv2.threshold(img,thresh_shadow,255,cv2.THRESH_BINARY_INV)
	img_shadow = cv2.bitwise_and(img_shadow,vignette_mask)
	_,img_highlight = cv2.threshold(img,thresh_highlight,255,cv2.THRESH_BINARY)
	img_sum = img_shadow + img_highlight
	img_stack = np.hstack((img_shadow, img_highlight, img_sum))
	cv2.imshow("image",img_stack)
	cv2.waitKey(5000)