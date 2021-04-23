import cv2
import numpy as np
from PIL import Image

Face_Detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Recognizer = cv2.face.LBPHFaceRecognizer_create()
Recognizer.read('trainer.yml')

Users = {1: "Taylor", 2: "Selena"}


def load_image(path):
    grayscale_pil_image = Image.open(path).convert('L')
    grayscale_arr_image = np.array(grayscale_pil_image, 'uint8')
    return grayscale_arr_image


def detect_face(image):
    faces = Face_Detector.detectMultiScale(image, 1.5, 3)
    print(faces)
    profiles = []
    for (x, y, w, h) in faces:
        index, dist = Recognizer.predict(image[y: y + h, x: x + w])

        print(index)
        print(dist)

        if dist <= 25:
            continue

        profiles.append(Users[index])

    return profiles


def __main__():
    image = load_image('test/1.jpg')
    profiles = detect_face(image)
    print(profiles)


__main__()
