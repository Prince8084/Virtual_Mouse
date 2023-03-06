import cv2
import mediapipe as mp
import pyautogui
from cvzone.HandTrackingModule import HandDetector

vid = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

vid.set(3, 1280)
vid.set(4, 720)
detector = HandDetector(detectionCon=0.8)

startDist = None
scale = 0
cx, cy = 500, 500

while True:

    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    success, img = vid.read()
    handss, img = detector.findHands(img)

    if hands:
        if len(handss) == 2:
            img1 = cv2.imread('pikatchu.jpeg')

            # print(detector.fingersUp(hands[0]), detector.fingersUp(hands[1])) #shows values for both right and left hands
            if detector.fingersUp(handss[0]) == [1, 1, 0, 0, 0] and detector.fingersUp(handss[1]) == [1, 1, 0, 0, 0]:
                # print("Zoom Gesture")
                lmList1 = handss[0]["lmList"]
                lmList2 = handss[1]["lmList"]

                # point 8 is the tip of the index finger
                if startDist is None:
                    # lmList1[8],lmList2[8]
                    # length, info, img = detector.findDistance(lmList1[8],lmList2[8],img)
                    length, info, img = detector.findDistance(handss[0]["center"], handss[1]["center"], img)

                    startDist = length

                # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)

                length, info, img = detector.findDistance(handss[0]["center"], handss[1]["center"], img)
                scale = int((length - startDist) // 2)
                cx, cy = info[4:]
                print(scale)
        else:
            startDist = None
        try:
            h1, w1, _ = img1.shape
            newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
            img1 = cv2.resize(img1, (newW, newH))

            img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1
        except:
            pass



        else:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand)
                landmarks = hand.landmark
                for id, landmarks in enumerate(landmarks):
                    x = int(landmarks.x * frame_width)
                    y = int(landmarks.y * frame_height)

                    if id == 8:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                        index_x = screen_width / frame_width * x
                        index_y = screen_height / frame_height * y
                        pyautogui.moveTo(index_x, index_y)
                    if id == 4:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                        thumb_x = screen_width / frame_width * x
                        thumb_y = screen_height / frame_height * y
                        print('outside', abs(index_y - thumb_y))
                        if abs(index_y - thumb_y) < 20:
                            pyautogui.click()
                            pyautogui.sleep(1)

    usb_camera = cv2.imshow('Virtual Mouse', frame)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        vid.release()

        cv2.destroyAllWindows()
