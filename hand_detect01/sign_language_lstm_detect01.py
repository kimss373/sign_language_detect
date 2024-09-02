from tensorflow import keras
import numpy as np
import cv2
import mediapipe as mp
import numpy.linalg as LA

seq_length = 5
seq = []

gesture = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F',
    6:'G', 7:'H', 8:'I', 9:'J'
}

model = keras.models.load_model('C://ai_project01/hand_lstm_train_result')
model.summary()

cv2.namedWindow(winname='webcam_window01', flags=cv2.WINDOW_NORMAL)

cv2.resizeWindow(winname='webcam_window01', width=1024, height=800)

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=500) as hands:
    while cap.isOpened() == True:
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if success == False:
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )
                joint = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],:]

                v = v2 - v1
                v_normal = LA.norm(v, axis=1)

                v_normal2 = v_normal[:, np.newaxis]
                v2 = v / v_normal2
                a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
                b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]
                ein = np.einsum('ij,ij->i', a, b)
                radian = np.arccos(ein)
                angle = np.degrees(radian)
                data = np.concatenate([joint.flatten(), angle])
                seq.append(data)
                if len(seq) < 5:
                    continue

                last_seq = seq[-5: ]
                input_arr = np.array(last_seq, dtype=np.float32)
                input_lstm_arr = input_arr.reshape(1, 5, 78)
                print(input_lstm_arr.shape)

                y_pred = model.predict(input_lstm_arr)
                idx = int(np.argmax(y_pred))
                letter = gesture[idx]
                conf = y_pred[0, idx]
                cv2.putText(image,
                            text=f"{letter} {round(conf*100,2)} percent!!",
                            org=(
                                int(hand_landmarks.landmark[0].x * image.shape[1]),
                                int(hand_landmarks.landmark[0].y * image.shape[0])
                            ),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2
                )

        cv2.imshow('webcam_window01', image)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()