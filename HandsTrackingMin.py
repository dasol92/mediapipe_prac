import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

if not cam.isOpened():
    print("Could not open cam")
    exit()

while cam.isOpened():
	success, img = cam.read()
	h, w, c = img.shape
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRGB)
	# print(results.multi_hand_landmarks)
	
	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			for id, lm in enumerate(handLms.landmark):
				# print(id, lm)
				cx, cy = int(lm.x * w), int(lm.y * h)
				if id == 4: # Thumb
					cv2.circle(img, (cx, cy), 30, (255, 0, 255), cv2.FILLED)


			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

	currTime = time.time()
	fps = 1/(currTime - prevTime)
	prevTime = currTime

	cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

	cv2.imshow("Image", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cam.release()
		cv2.destroyAllWindows()
		break
