import os
import pickle

import cv2
import face_recognition
from imutils import paths

from config import settings


class EncodeFaces:

    dataset = 'dataset'
    encodings = settings.encodings

    def __init__(self, detection_method='hog'):
        self.detection_method = detection_method


    def encode_faces(self):
        try:
            print("[INFO] Quantifying faces...")
            imagePaths = list(paths.list_images(self.dataset))

            knownEncodings = []
            knownNames = []

            for (i, imagePath) in enumerate(imagePaths):
                print("[INFO] Processing image {}/{}".format(i + 1, len(imagePaths)))
                name = imagePath.split(os.path.sep)[-2]

                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                boxes = face_recognition.face_locations(rgb, model=self.detection_method)

                encodings = face_recognition.face_encodings(rgb, boxes)

                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)
                    
        except Exception as exc:
            print("[ERROR] Something happened...")
            print(exc)

        else:
            print("[INFO] Serializing encodings...")

            data = {"encodings": knownEncodings, 
                    "names": knownNames}
            with open(self.encodings, "wb") as f:
                f.write(pickle.dumps(data))

            print("[INFO] Completed...")