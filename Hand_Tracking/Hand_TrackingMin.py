import cv2
import mediapipe as mp
import  time

cap = cv2.VideoCapture(0)

# create an object to track or detection
mpHands = mp.solutions.hands

# hands is an object
hands= mpHands.Hands()

# to draw line between  we use a math function
mpDraw = mp.solutions.drawing_utils

# frame rate displaying
cTime = 0
pTime =0

while True:
    success, img = cap.read()

    # we have to convert image BGR to RGB  beacuse it only accept RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # keep that RGB image into object which we have crrated
    results = hands.process(imgRGB)
    # to check something is detected or not
    # print(results.multi_hand_landmarks)

    # to extract multiple hands and extract them one by one
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, lms in enumerate(handLandmarks.landmark):
                # print(id,lms)
                height, width, channels = img.shape
                cx, cy = int(lms.x*width), int(lms.y*height)
                print(id, cx, cy)

                # cx,cy information
                # if id==0:
                #     cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # to display on the screen framerate
    cv2.putText(img, str(int(fps)), (10,78), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)