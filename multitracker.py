import cv2, os, time
import numpy as np

# path to folder with our script
path = os.getcwd()
# creating MultiTracker object
tracker = cv2.MultiTracker_create()

# select video
vid = cv2.VideoCapture(path + '/video/Circle_illusion.mp4')
#vid = cv2.VideoCapture(path + '/video/vids_1.mp4')

# start 1st frame to choose objects
working, frame = vid.read()

# select objects to follow
bboxes, colors = [],[]
while (True):
    # draw bounding boxes
    bbox = cv2.selectROI('MultiTracker', frame)
    # adding selected object to array
    bboxes.append(bbox)
    # color the object (random color)
    colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))

    print("Press 'q' to quit selecting boxes and start tracking\nPress 'Enter' to select more objects")

    k = cv2.waitKey(0) & 0xFF
    if (k == 113):
        break

# initialize MultiTracker
for bbox in bboxes:
    # adding tracking algorithms

    # CSRT - usefull, but slow when 2+ objects
    # tracker.add(cv2.TrackerCSRT_create(), frame, bbox)

    # BOOSTING - goes perfrct
    tracker.add(cv2.TrackerBoosting_create(), frame, bbox)

# videostreaming and tracking objects
while (vid.isOpened()):
    # read stream
    working, frame = vid.read()

    # check if video is over
    if (not working):
        print('End of video')
        break

    # update location of the objects
    working, boxes = tracker.update(frame)
    # drawing objects
    for i, newbox in enumerate(boxes):
        # choose coordinates
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        # set rectangle to the object
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        # get coordinates of the label
        (x, y) = p1
        # set color for label
        cv2.putText(frame,'{}'.format(i + 1), (x+5, y+20), cv2.FONT_HERSHEY_PLAIN, fontScale = 2,color = (0,0,0), thickness = 2);

    # open window and show video
    cv2.imshow('MultiTracker', frame)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
