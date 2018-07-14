# Identify Eye Gaze for Autism Screening- Eye Contact
import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 640,480
w = 650
h = 450

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        # sample
        frameD = cv2.pyrDown(cv2.pyrDown(frame))
        frameDBW = cv2.cvtColor(frameD,cv2.COLOR_RGB2GRAY)

        # detect eye through openCv library
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = cv2.CascadeClassifier('lib/haarcascade_eye.xml')
        detected = faces.detectMultiScale(frame, 1.3, 5)

        # detect eye through openCv library
        faces = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
        detected2 = faces.detectMultiScale(frameDBW, 1.3, 5)

        # Small Square
        pupilFrame = frame
        pupilO = frame
        windowClose = np.ones((5, 5), np.uint8)
        windowOpen = np.ones((2, 2), np.uint8)
        windowErode = np.ones((2, 2), np.uint8)

        # Draw square
        for (x, y, w, h) in detected:
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 1)
            cv2.line(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.line(frame, (x + w, y), (x, y + h), (0, 0, 255), 1)
            pupilFrame = cv2.equalizeHist(frame[int(y + (h * .25)):int((y + h)), x:(x + w)])
            # pupilO = pupilFrame
            et, pupilFrame = cv2.threshold(pupilFrame, 150, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
            # 50-70 is better
            pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
            pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
            pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)

            # set the gaze into the blob
            threshold = cv2.inRange(pupilFrame, 150, 255)
            # contours, hierarchy = cv2.findContours(threshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Working till here
            print(contours)

            # There are 1 or less blobs, do nothing and have 2 blobs
            # Not defined the logic correctly
            if len(contours) <= 2:
                # find biggest blob
                maxArea = 0
                MAindex = 0  # to get the unwanted frame
                distanceX = []  # delete the left most (for right eye)
                currentIndex = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    center = cv2.moments(cnt)
                    # print(area)
                    cx, cy = int(center['m10'] / center['m00']), int(center['m01'] / center['m00'])
                    distanceX.append(cx)
                    if area > maxArea:
                        maxArea = area
                        MAindex = currentIndex
                    currentIndex = currentIndex + 1

                del contours[MAindex]  # remove the picture frame contour
                del distanceX[MAindex]

            eye = 'right'

            # delete the left most blob for right eye
            if len(contours) <= 2:
                if eye == 'right':
                    edgeOfEye = distanceX.index(min(distanceX))
                else:
                    edgeOfEye = distanceX.index(max(distanceX))
                del contours[edgeOfEye]
                del distanceX[edgeOfEye]

            # get largest blob
            if len(contours) <= 1:
                maxArea = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    print(area)
                    if area > maxArea:
                        maxArea = area
                        largeBlob = cnt

            # if len(largeBlob) > 0:
            #     center = cv2.moments(largeBlob)
            #     cx, cy = int(center['m10'] / center['m00']), int(center['m01'] / center['m00'])
            #     cv2.circle(pupilO, (cx, cy), 5, 255, -1)

        # show picture
        cv2.imshow('frame', pupilO)
        cv2.imshow('frame2', pupilFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# else:
# break

# Release the objects
cap.release()
cv2.destroyAllWindows()
