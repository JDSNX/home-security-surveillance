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

    def __init__(self, video_channel=0, output='output/video.avi', detection_method='hog'):
        self.output = output
        self.video_channel = video_channel
        self.detection_method = detection_method
        self.authorize_output = 'output/authorize'
        self.unauthorize_output = 'output/unauthorize'

        is_ready("face-recognized", True)


    async def face_recognize(self):
        cap = cv2.VideoCapture(self.video_channel)

        while True:
            ret, frame = cap.read()

            if ret is False:
                print('[ERROR] Something wrong with your camera...')
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = frame.shape[1] / float(rgb.shape[1])

            boxes = face_recognition.face_locations(rgb, model=self.detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            for ((top, right, bottom, left), encoding) in zip(boxes, encodings):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                name, color = "Unknown", (0, 0, 255)

                matches = face_recognition.compare_faces(self.data["encodings"], encoding)
                
                face_distances = face_recognition.face_distance(self.data["encodings"], encoding)

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.data["names"][best_match_index]
                    color = (0, 255, 0)

                s = cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
                cv2.putText(s, name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                if name == "Unknown":
                    await detected("face-recognized", False, name)
                    cv2.imwrite(f'{self.unauthorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg', frame)
                else:
                    await detected("face-recognized", True, name)
                    cv2.imwrite(f'{self.authorize_output}/{datetime.now().strftime("%d_%m_%Y_%H_%M")}_{name}.jpg', frame)

            cv2.imshow("Face Recognition", frame)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        await is_ready("face-recognized", False)
        await detected("face-recognized", False, name)
        cap.release()
        cv2.destroyAllWindows()