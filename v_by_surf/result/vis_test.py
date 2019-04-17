# read test.txt and check the results with video
import numpy as np
import cv2
import csv

train_end = 2000

pred = []
with open('test_3.txt',encoding='utf-8') as cfile:
			reader = csv.reader(cfile)
			readeritem=[]
			readeritem.extend([row for row in reader])
for i, row in enumerate(readeritem):
	pred.append(float(row[0]))
del reader
del readeritem

cap = cv2.VideoCapture('../../data/test.mp4')
out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
f = 0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.putText(frame,'v=',(20,100),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,255,0), 2)
    cv2.putText(frame, str(pred[f])[0:5],(90,100),cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,255,0), 2)
    cv2.putText(frame, '@frame:'+str(f)[0:5],(20,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 1)
    cv2.line(frame, (220, 260), (220+200, 260), (0,0,255), 1)
    cv2.line(frame, (44, 260+80), (44+552, 260+80), (0,0,255), 1)
    cv2.line(frame, (220, 260), (44, 260+80), (0,0,255), 1)
    cv2.line(frame, (220+200, 260), (44+552, 260+80), (0,0,255), 1)

    # Display the resulting frame
    cv2.imshow('Video',frame)
    out.write(frame)
    f+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if f==train_end+1:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()