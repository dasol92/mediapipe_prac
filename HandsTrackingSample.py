import cv2
import HandsTrackingModule as htm
import time

prevTime = 0
currTime = 0

cam = cv2.VideoCapture(0)
if not cam.isOpened():
	print("Could not open cam")
	exit()

detector = htm.handDetector()

while cam.isOpened():
	success, img = cam.read()
	img = detector.findHands(img)
	lmList = detector.findPosition(img)
	if len(lmList) != 0:
		print(lmList[4]) # Thumb

	currTime = time.time()
	fps = 1/(currTime - prevTime)
	prevTime = currTime

	cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

	cv2.imshow("Image", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cam.release()
		cv2.destroyAllWindows()
		break