import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands() as hands:
    while cap.isOpened()== True:

        success, image = cap.read()

        image = cv2.flip(image, 1)

        if success == False:
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks != None:
            cv2.putText(
                image,
                text="Detect Hand!!",
                org=(300, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0,0,255),
                thickness=2
            )

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
                #print(hand_landmarks.landmark[12].x)
                if hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y:
                    cv2.putText(
                        image,
                        text="Open!!",
                        org=(int(hand_landmarks.landmark[12].x*640), int(hand_landmarks.landmark[12].y*480)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2
                    )
                else:
                    cv2.putText(
                        image,
                        text="Close!!",
                        org=(int(hand_landmarks.landmark[12].x*640), int(hand_landmarks.landmark[12].y*480)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2
                    )




        cv2.imshow('webcam_window01', image)

        if cv2.waitKey(1) == ord('q'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imsave("cam_img.jpg", image)
            break

cap.release()