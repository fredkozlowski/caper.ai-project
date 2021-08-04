from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import statistics

def detectBarcode(frame):
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
	grayFrame = clahe.apply(grayFrame)
	edgeEnchancement =  cv2.Laplacian(grayFrame, ddepth = cv2.CV_8U, ksize = 3, scale = 1, delta = 0)
	blur = cv2.bilateralFilter(edgeEnchancement, 13, 50, 50)
	(_, thresh) = cv2.threshold(blur, 55, 255, cv2.THRESH_BINARY)

	# isolate barcode and get contours
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	closed = cv2.erode(closed, None, iterations = 4)
	closed = cv2.dilate(closed, None, iterations = 4)
	(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	try:
		c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
		rect = cv2.minAreaRect(c)
		box = np.int0(cv2.boxPoints(rect))
		
		copy = frame.copy()
		cv2.drawContours(copy, [box], -1, (0, 255, 0), 3)
		# print(box)
		retval = cv2.imwrite("found.jpg", copy)

		maxHeight, maxWidth, _ = frame.shape
		output_pts = np.float32([[0, 0],
	                        [0, maxHeight - 1],
	                        [maxWidth - 1, maxHeight - 1],
	                        [maxWidth - 1, 0]])
		M = cv2.getPerspectiveTransform(np.float32(box) ,output_pts)
		warped = cv2.warpPerspective(frame,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
		# uncomment these lines to see the detected barcode 
		# cv2.imshow("warped", warped)
		# cv2.waitKey(0)
		retval = cv2.imwrite("warped.jpg", warped)
	except IndexError:
		print("barcode not detected properly")
	# here I would add a barcode reading function using pyzbar
	# unfortunately my computer can't install it due to an old version of MacOS
	# so instead this code always return that no barcode was read
	return -1


# if object exits frame and bounding box center was trending downwards
# then we can assume it's been put into the cart
# this is the list of bounding box positions
itemPositions = []

# path to the labels
LABELS = open("yolo-coco/coco.names").read().strip().split("\n")

# colors for bounding boxes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# setting up darknet and getting output layers
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# setting up video stream
vs = VideoStream(src=0).start()
time.sleep(.5)
writer = None
(W, H) = (None, None)

objectLocationList = []
lastPosition = 0
noObject = -1

objectFromBarcode = ""

while True:
	frame = vs.read()

	if detectBarcode(frame) != -1:
		objectFromBarcode = "bottle" # this is where I would convert an SKU to an object

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame 
	# apply a forward pass to the blob
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > .5:
				# YOLO returns the center of the bounding box, as well
				# as the width, height - converting to top left
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# if a bottle is detected, append
				# the relevant data for later processing
				if LABELS[classID] == "bottle":
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
					objectLocationList.append(centerY)

	# apply non max suppression
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
		0.3)

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw the bounding box
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			noObject = 0
	else:
		# check how many frames since a detected object was last detected
		if noObject != -1:
			noObject += 1
			print(noObject)

	# check if the object has been put into basket
	# a quick and easy heuristic to determine whether an object has 
	# entered the basket is if it has been gone for a few frames 
	# and the last known position was lower than the median of
	# all positions
	if noObject > 4 and objectLocationList[-1] > statistics.median(objectLocationList):
		cv2.putText(frame, "added to cart", (200, 200),
				cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
		noObject = -1
		if objectFromBarcode != "":
			print("added " + objectFromBarcode + " to cart")
		else:
			print("added bottle to cart")
		objectFromBarcode = ""
	elif noObject > 4:
		noObject = -1

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
	writer.write(frame)



writer.release()
vs.release()




