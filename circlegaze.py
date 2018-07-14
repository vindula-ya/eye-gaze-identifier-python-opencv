import imutils
import numpy as np
import cv2

# Read Image Sorce
image1 = cv2.imread('src/baby1.jpg')
image2 = cv2.bitwise_not(image1)

kernel = np.ones((4,3),np.uint8)
# sure background area
sure_bg = cv2.erode(image2,kernel,iterations=3)

# sure_bg = cv2.erode(image1,kernel,iterations=3)
image = cv2.dilate(sure_bg,kernel,iterations=3)
# image = cv2.bitwise_not(sure_bg2)
image = image1

gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((3,3),np.uint8)
# sure background area
sure_bg = cv2.dilate(thresh,kernel,iterations=3)
imageA = cv2.erode(sure_bg,kernel,iterations=8)
imagenot = cv2.bitwise_not(imageA)

# find contours in the thresholded image
cnts = cv2.findContours(imagenot.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
loaction = list()
# print('start')
# loop over the contours
i = 0
for c in cnts:
	i = i +1
	# compute the center of the contour
	# print('loop')
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# draw the contour and center of the shape on the image
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
	cv2.putText(image, "center", (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	# global loaction
	loaction.append([cX, cY])

	# print(loaction)
	# print ("x coordinate Number: ")
	# print (cX)
	# print ("y coordinate Number: ")
	# print (cY)



cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

