import cv2
import numpy as np
from PIL import Image
import os


def training():

    # pastas das imagens do id
    path = os.path.dirname(os.path.abspath(__file__)) + '/dataset'

    # HINTOGRAMAS DE PADRÕES BINÁRIOS LOCAIS
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # escala cinza
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

        return faceSamples, ids

    print("\n Treinando rostos, por favor aguarde ...")
    # pegas as imagens do dataset e retorna 2 matrizes: "IDS" e "Faces"
    faces, ids = getImagesAndLabels(path)

    recognizer.train(faces, np.array(ids))

    # Salvando o modelo em trainer/trainer.yml
    recognizer.write(f'{path}/../trainer/trainer.yml')

    # Print the numer of faces trained and end program
    print("\n {0} faces treinadas. Saindo do programa!".format(
        len(np.unique(ids))))
