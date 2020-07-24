import cv2
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))

#treinando o algoritmo de reconhecimento

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(f'{path}/trainer/trainer.yml')

cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['', 'Rafael', 'meu amo', 'tonice']

cam = cv2.VideoCapture(0)
cam.set(3, 800)
cam.set(4, 600)

#Definir o tamanho minimo da janela a ser reconhecido como uma face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH))
    )
    for(x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence > 50:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            ########################

            print(id)

            id = names[id]
            confidence = " {0}%".format(round(confidence))
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            id = "Sem registro"
            confidence = " {0}%".format(round(confidence))

        cv2.putText(
            img,
            str(id),
            (x+5, y-5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h-5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n Saindo do programa e limpando")
cam.release()
cv2.destroyAllWindows()