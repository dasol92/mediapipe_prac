import cv2
import mediapipe as mp
import time

class handDetector():
	def __init__(self, static_image_mode=False, 
				max_num_hands=2, model_complexity=1,
				min_detection_confidence=0.5, min_tracking_confidence=0.5):
		self.mode = static_image_mode
		self.maxHands = max_num_hands
		self.modelComplexity = model_complexity
		self.min_detection_confidence = min_detection_confidence
		self.min_tracking_confidence = min_tracking_confidence

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.min_detection_confidence, self.min_tracking_confidence)
		self.mpDraw = mp.solutions.drawing_utils

	def findHands(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(imgRGB)		
		if self.results.multi_hand_landmarks:
			for handLms in self.results.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

		return img

	def findPosition(self, img, handNo=0, draw=True):
		h, w, c = img.shape
		lmList = []
		if self.results.multi_hand_landmarks:
			myHand = self.results.multi_hand_landmarks[handNo]
			for id, lm in enumerate(myHand.landmark):
				cx, cy = int(lm.x * w), int(lm.y * h)
				lmList.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
		
		return lmList

def main():
	prevTime = 0
	currTime = 0

	cam = cv2.VideoCapture(0)
	if not cam.isOpened():
		print("Could not open cam")
		exit()

	detector = handDetector()

	while cam.isOpened():
		success, img = cam.read()
		img = detector.findHands(img)
		lmList = detector.findPosition(img)
		if len(lmList) != 0:
			print(lmList[4])

		currTime = time.time()
		fps = 1/(currTime - prevTime)
		prevTime = currTime

		cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

		cv2.imshow("Image", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cam.release()
			cv2.destroyAllWindows()
			break

if __name__ == "__main__":
	main()