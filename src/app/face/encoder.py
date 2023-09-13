import os
from pickle import dumps

import cv2
from face_recognition import face_locations, face_encodings
from imutils import paths

from config import settings, logger


class EncodeFaces:

    dataset = 'dataset'
    encodings = settings.encodings

    def __init__(self, detection_method='hog'):
        self.detection_method = detection_method


    def encode_faces(self):
        try:
            logger.info("Quantifying faces...")
            imagePaths = list(paths.list_images(self.dataset))

            knownEncodings = []
            knownNames = []

            for (i, imagePath) in enumerate(imagePaths):
                logger.info("Processing image {}/{}".format(i + 1, len(imagePaths)))
                name = imagePath.split(os.path.sep)[-2]

                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                boxes = face_locations(rgb, model=self.detection_method)

                encodings = face_encodings(rgb, boxes)

                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)
                    
        except Exception as exc:
            logger.error("Something happened...")
            logger.info(exc)
            

        else:
            logger.info("Serializing encodings...")

            data = {"encodings": knownEncodings, 
                    "names": knownNames}
            with open(self.encodings, "wb") as f:
                f.write(dumps(data))

            logger.info("Completed...")