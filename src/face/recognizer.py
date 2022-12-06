import time
import face_recognition
import pickle
import cv2
import numpy as np
from datetime import datetime

from modules import detected, is_ready

class FaceRecognition:

    encodings='encodings.pickle'
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    data = pickle.loads(open(encodings, "rb").read())
    recognize = 'output/authorize'
    unrecognize = 'output/unauthorize'

    async def __init__(self, video_channel=0, output='output/video.avi', detection_method='hog'):
        self.output = output
        self.video_channel = video_channel
        self.detection_method = detection_method
        self.authorize_output = 'output/authorize'
        self.unauthorize_output = 'output/unauthorize'

        await is_ready("face-recognized", True)


    async def face_recognize(self):
        cap = cv2.VideoCapture(self.video_channel)

        boxes = []
        encodings = []
        names = []
        process_this_frame = True

        while True:
            ret, frame = cap.read()

            if ret is False:
                print('[ERROR] Something wrong with your camera...')
                break

            if process_this_frame:
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

                rgb_small_frame = small_frame[:,:,::-1]

                boxes = face_recognition.face_locations(rgb_small_frame)
                encodings = face_recognition.face_encodings(rgb_small_frame, boxes)

                names = []
                for encoding in encodings:
                    matches = face_recognition.compare_faces(self.data["encodings"], encoding)
                    name = "Unknown"
                    
                    face_distances = face_recognition.face_distance(self.data["encodings"], encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.data["names"][best_match_index]

                    names.append(name)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(boxes, names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.putText(frame, name, (left, top - 5), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            
            if name == "Unknown":
                #await detected("face-recognized", False, name)
                cv2.imwrite(f'{self.unauthorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg', frame)
            else:
                #await detected("face-recognized", True, name)
                cv2.imwrite(f'{self.authorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg', frame)
            
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        await is_ready("face-recognized", False)
        await detected("face-recognized", False, name)
        cap.release()
        cv2.destroyAllWindows()