import cv2
import os
import FaceTraining as faceTraining

path = os.path.dirname(os.path.abspath(__file__))
cam = cv2.VideoCapture(0)
isNotRegister = True

cam.set(3, 640)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_id = input('\n insira um id de usuário e pessione <enter> ==>  ')
face_name = input('\n insira um nome de usuário e pessione <enter> ==>  ')


def storageUser():
    global isNotRegister, face_id

    f = open(f'{path}/names.txt', 'r')
    for line in f:
        if face_id == line[0]:
            isNotRegister = False
            break
        else:
            isNotRegister = True

    if isNotRegister:
        with open(f'{path}/names.txt', 'a') as outFile:
            outFile.write(f'{face_id} - {face_name}  \n')
        register()
    else:
        print('\n Id de usuário já cadastrado!')


def register():
    print("\n iniciando captura de rosto. Olhe a câmera e aguarde ...")

    count = 1

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite(f"{path}/dataset/User." + str(face_id) + '.' +
                        str(count) + ".jpg", gray[y:y+h, x:x+w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video

        if k == 27:
            break
        elif count >= 30:
            break

    print("\n fechando. Próximo passo! aguarde...")
    cam.release()
    cv2.destroyAllWindows()
    faceTraining.training()


if face_id != '' and face_name != '':
    storageUser()
else:
    print('Campos não preenchidos corretamente')