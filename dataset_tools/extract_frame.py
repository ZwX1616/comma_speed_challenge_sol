# convert the video frames into trainable images

# training set
# import cv2
# vc = cv2.VideoCapture('../data/train.mp4')

# count = 0

# isread, frame = vc.read()
# while isread:
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = frame[270:270+80,200:200+240]
#     frame = cv2.resize(frame,(192,64))
# #    import pdb; pdb.set_trace()
#     cv2.imwrite('../data/image_train/'+str(count)+'.jpg', frame)
#     print('written frame '+str(count), end='\r')
#     isread, frame = vc.read()
#     count += 1

# print('finish reading train.mp4')

# test set
import cv2
vc = cv2.VideoCapture('../data/test.mp4')

count = 0

isread, frame = vc.read()
while isread:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[270:270+80,200:200+240]
    frame = cv2.resize(frame,(192,64))
#    import pdb; pdb.set_trace()
    cv2.imwrite('../data/image_test/'+str(count)+'.jpg', frame)
    print('written frame '+str(count), end='\r')
    isread, frame = vc.read()
    count += 1

print('finish reading test.mp4')