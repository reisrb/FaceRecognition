import numpy as np
import cv2
import os

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #carregando os classificadores de detectação do rosto em uma imagem
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') #olhos
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml') #boca


cap = cv2.VideoCapture(0) #pegando a camera principal e retornando a variável
cap.set(3, 800) #exibindo em um tamanho especificado
cap.set(4, 600)

while True:
    ret, img = cap.read() #pegando a captura
    img = cv2.flip(img, 1) #deixando na posição vertical normal
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #a imagem em escala cinza para detecção
    faces = faceCascade.detectMultiScale(gray, 1.2, 3) #função classificadora: fator de escala, número de vizinhos e tamanho mínimo da face detectada = minSize = (20, 20).

    for (x, y, w, h) in faces: #detectar rostos na imagem marcando cada face com um retangulo azul
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3) #onde pegando a posição da largura e altura da minha imagem será setada o retangulo dando coordenadas x,y
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smile = smileCascade.detectMultiScale(roi_gray)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.imshow('video', img) #criando uma janela e mostrando na tela

    k = cv2.waitKey(30) & 0xff #tecla esc para sair
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows() #fecha