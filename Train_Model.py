import cv2
import os
import numpy as np
from PIL import Image
import cv2

Recognizer = cv2.face.LBPHFaceRecognizer_create()
Detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def get_image(path):
    user_images = {}
    for user_dir in os.listdir(path):  # User_dir is also user index
        user_images[user_dir] = []
        for image_file_name in os.listdir(os.path.join(path, user_dir)):
            if image_file_name[-3:] == 'jpg':
                image_path = os.path.join(path, user_dir, image_file_name)
                grayscale_pil_image = Image.open(image_path).convert('L')

                # Convert to numpy array
                grayscale_arr_image = np.array(grayscale_pil_image, 'uint8')

                user_images[user_dir].append(grayscale_arr_image)
    return user_images


def get_face_samples(images):
    face_samples = []
    ids = []
    for key in images.keys():
        for image in images[key]:
            faces = Detector.detectMultiScale(image)
            for (x, y, w, h) in faces:
                face_samples.append(image[y: y + h, x: x + w])
                ids.append(int(key))

    return face_samples, ids


def __main__():
    images = get_image('dataSet')
    face_samples, ids = get_face_samples(images)

    Recognizer.train(face_samples, np.array(ids))
    Recognizer.save('trainer.yml')


__main__()
