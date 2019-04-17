import cv2

prev = cv2.imread('../data/image_train/'+str(0)+'.jpg')
next = cv2.imread('../data/image_train/'+str(0+1)+'.jpg')

tracker = cv2.TrackerGOTURN_create()

bbox = cv2.selectROI(prev, False)

ok = tracker.init(prev,bbox)

ok, bbox = tracker.update(next)

if ok:
    # Tracking success
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(next, p1, p2, (255,0,0), 2, 1)

cv2.imshow("Tracking", next)
cv2.waitKey(8000)