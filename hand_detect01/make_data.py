from tkinter import *

import cv2
import mediapipe as mp

import numpy as np
import os
import numpy.linalg as LA

os.makedirs('hand_dataset', exist_ok=True)
frame=0
MAX_FRAME = 600
all_data = []

action = "미정"

def btnpress():
    global action
    print('버튼 클릭했음!!!')
    input = ent.get()
    print("input=", input)
    action = input
    window.destroy()

window = Tk()

ent = Entry(window)
ent.pack()

label = Label(window)
label.config(text="데이터 입력할 알파벳을 입력하세요")

label.pack()

btn = Button(window)
btn.config(text="확인")
btn.config(command=btnpress)
btn.pack()

window.mainloop()

print("action=", action)

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands() as hands:

    while cap.isOpened() == True:

        frame = frame + 1
        if frame >= MAX_FRAME:
            break

        success, image = cap.read()

        image = cv2.flip(image, 1)

        if success == False:
            continue
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks!=None:
            cv2.putText(
                image,
                text="Detect Hand!!",
                org=(300, 50),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2
            )
            cv2.putText(
                image,
                text=f"Gathering {action} Data Frame: {MAX_FRAME - frame} Left",
                org=(0, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
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
                joint = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    print("j=",j)
                    print("lm=",lm)
                    print("lm.x=",lm.x)
                    print("lm.y=",lm.y)
                    print("lm.z=",lm.z)
                    joint[j] = [lm.x, lm.y, lm.z]
                    print("="*100)
                print("joint=",joint)
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = v2-v1
                print("="*100)
                print("v=",v)
                print("="*100)

                v_normal = LA.norm(v, axis=1)
                print("="*100)
                print("v_normal=",v_normal)
                print("="*100)

                v_normal2 = v_normal[:, np.newaxis]
                print("v_normal2=",v_normal2)

                v2 = v / v_normal2
                print("="*100)
                print("v2=", v2)
                print("="*100)

                a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18],:]
                b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]
                ein = np.einsum('ij,ij->i', a, b)
                print("="*100)
                print("ein=",ein)
                print("="*100)

                radian = np.arccos(ein)
                print("radian=", radian)

                angle = np.degrees(radian)
                print("angle=",angle)

                action_num = ord(action)
                print("="*100)
                print("action_num=", action_num)
                print("="*100)

                action_label = action_num - ord('A')
                print("="*100)
                print("action_label=",action_label)
                print("="*100)

                angle_label = np.append(angle, action_label)
                print("="*100)
                print("angle_label=", angle_label)
                print("="*100)

                data = np.concatenate([joint.flatten(), angle_label])

                print("="*100)
                print("joint=", joint)
                print("="*100)
                print("joint.flatten()=", joint.flatten())
                print("="*100)
                print("data=", data)
                print("="*100)

                all_data.append(data)

        cv2.imshow('webcam_window01', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()

import time

created_time = int(time.time())

np.save(os.path.join('./hand_dataset', f'{action}_{created_time}'), all_data)