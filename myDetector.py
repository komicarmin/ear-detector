import operator

import numpy, cv2, sys, os

cascadeRightEar = cv2.CascadeClassifier("haarcascade_mcs_rightear.xml")
cascadeLeftEar = cv2.CascadeClassifier("haarcascade_mcs_leftear.xml")
cascadeHeadFront = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
cascadeHeadSide = cv2.CascadeClassifier("haarcascade_profileface.xml")
cascadeHeadAndSholders = cv2.CascadeClassifier("HS.xml")

inDir = "./AWEForSegmentation/train"
outDir = "./myOut/"
annotRectDir = "./AWEForSegmentation/trainannot_rect"

def detectLeftEar(img, annotImg, scaleFactor, minNeighbors, validate = False):
	detectionList = list(cascadeLeftEar.detectMultiScale(img, scaleFactor, minNeighbors))
	if validate:
		validatedList = []
		#bbs = getBoundingBoxes(annotImg)
		for detection in detectionList:
			# valid = validateDetection(bbs, detection)
			valid = validateColor(img, detection)
			validatedList.append([detection, valid])
		detectionList = validatedList
	return detectionList

def detectRightEar(img, annotImg, scaleFactor, minNeighbors, validate = False):
	detectionList = list(cascadeRightEar.detectMultiScale(img, scaleFactor, minNeighbors))
	if validate:
		validatedList = []
		#bbs = getBoundingBoxes(annotImg)
		for detection in detectionList:
			# valid = validateDetection(bbs, detection)
			valid = validateColor(img, detection)
			validatedList.append([detection, valid])
		detectionList = validatedList
	return detectionList

def detectHeadSide(img, scaleFactor, minNeighbors):
	sideDetectionList = list(cascadeHeadSide.detectMultiScale(img, scaleFactor, minNeighbors))
	if len(sideDetectionList) > 0:
		return sorted(sideDetectionList, key=lambda x: x[2] * x[3], reverse=True)[0]
	return None

def detectHeadFront(img, scaleFactor, minNeighbors):
	detectionList = list(cascadeHeadFront.detectMultiScale(img, scaleFactor, minNeighbors))
	if len(detectionList) > 0:
		return sorted(detectionList, key=lambda x: x[2] * x[3], reverse=True)[0]
	return None

def detectHeadAndSholders(img, scaleFactor, minNeighbors):
	detectionList = list(cascadeHeadAndSholders.detectMultiScale(img, scaleFactor, minNeighbors))
	if len(detectionList) > 0:
		return sorted(detectionList, key=lambda x: x[2] * x[3], reverse=True)[0]
	return None

def intersection(a, b):
	x = max(a[0], b[0])
	y = max(a[1], b[1])
	w = min(a[0]+a[2], b[0]+b[2]) - x
	h = min(a[1]+a[3], b[1]+b[3]) - y
	return x, y, max(w, 0), max(h, 0)

def intersectionOverUnion(rect1, rect2):
	intersect = intersection(rect1, rect2)
	rect1Area = rect1[2] * rect1[3]
	rect2Area = rect2[2] * rect2[3]
	intersectionArea = intersect[2] * intersect[3]
	iou = intersectionArea / (rect1Area + rect2Area - intersectionArea)
	#print("IOU: " + str(iou))
	return iou

def getBoundingBoxes(image):
    LUV = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    edges = cv2.Canny(LUV, 10, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return list(map(cv2.boundingRect, contours))

def validateDetection(annots, detection):
	for bb in annots:
		if intersectionOverUnion(detection, bb) > 0.25:
			return True
	return False

def validateColor(img, region):
	x = region[0]
	y = region[1]
	w = region[2]
	h = region[3]
	section = img[y:y+h, x:x+w]
	section = section.reshape(h*w, 3)
	#bgrSum = [sum(i[0] for i in section), sum(i[1] for i in section), sum(i[2] for i in section)]
	#bgrAvg = list(map(lambda x: x/len(section), bgrSum))
	validPixels = sum([1 if x[2] > x[1] > x[0] else 0 for x in section])
	allPixels = len(section)
	percent = validPixels/allPixels * 100
	#print(percent)
	return percent >= 80

def writeToImage(img, detectionList1, detectionList2, dir, filename):
	for [(x, y, w, h), valid] in detectionList1:
		if valid:
			cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
		else:
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
	for [(x, y, w, h), valid] in detectionList2:
		if valid:
			cv2.rectangle(img, (x,y), (x+w, y+h), (255, 128, 0), 4)
		else:
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
	cv2.imwrite(os.path.join(dir, filename), img)

def process(scaleFactor, minNeighbors, visualize = False):
	ears = 0
	validEars = 0;
	maxEars = ["", 0]
	maxValidEars = ["", 0]
	files = os.listdir(inDir)
	for file in files:
		if file.endswith(".png"):
			print(file)
			img = cv2.imread(os.path.join(inDir, file))
			annotImg = cv2.imread(os.path.join(annotRectDir, file))

			leftEarDetectionList = detectLeftEar(img, annotImg, scaleFactor, minNeighbors, True)
			rightEarDetectionList = detectRightEar(img, annotImg, scaleFactor, minNeighbors, True)

			earsDetected = len(leftEarDetectionList) + len(rightEarDetectionList)
			ears += earsDetected
			if earsDetected > maxEars[1]:
				maxEars[0] = file
				maxEars[1] = earsDetected
			elif earsDetected == maxEars[1]:
				maxEars[0] += "," + file

			validEarsDetected = len(list(filter(lambda x: x[1] is True, leftEarDetectionList))) + len(list(filter(lambda x: x[1] is True, rightEarDetectionList)))
			validEars += validEarsDetected
			if validEarsDetected > maxValidEars[1]:
				maxValidEars[0] = file
				maxValidEars[1] = validEarsDetected
			elif earsDetected == maxEars[1]:
				maxValidEars[0] += "," + file

			if visualize:
				writeToImage(img, leftEarDetectionList, rightEarDetectionList, outDir, file)


	f = open(os.path.join(outDir, "details.txt"), "a")
	f.write("DETECTION DETAILS:\n")
	f.write("	scalefactor: " + str(scaleFactor) + "\n")
	f.write("	minNeighbors: " + str(minNeighbors) + "\n")
	f.write("	Images/faces: " + str(len(files)) + "\n")
	f.write("RESULTS:\n")
	f.write("	Ears: " + str(ears) + "\n")
	f.write("	Valid ears: " + str(validEars) + "\n")
	f.write("	Invadalid ears: " + str(ears - validEars) + "\n")
	f.write("	Max ears: " + str(maxEars[1]) + " in picture " + maxEars[0] + "\n")
	f.write("	Max valid ears: " + str(maxValidEars[1]) + " in picture " + maxValidEars[0] + "\n")
	f.write("##################################################\n")
	f.close()


if not os.path.exists(outDir):
	os.makedirs(outDir)

# for sf in [1.4, 1.3, 1.2, 1.1, 1.05, 1.01]:
# 	for mn in [1, 2, 3, 4, 5]:
# 		#print(sf, " - ", mn)
# 		process(sf, mn)

process(1.01, 5, True)

sys.exit()