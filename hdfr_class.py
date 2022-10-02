import datetime
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os
import numpy as np

from imutils import paths

from modules import YOLO_CFG, YOLO_WEIGHTS, detected, get_classes

class HumanDetection:
    def __init__(self, video_channel=None, roi=None, output_name=None):
        self.net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(320, 320), scale=1/255)

        self.output_name = f'output/{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}_output.avi' \
            if output_name is None else output_name
        self.video_channel = 0 if video_channel is None else video_channel
        self.classes = get_classes()
        self.roi = roi

    def check_intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y

        return False if w < 0 or h < 0 else True

    def detection(self):
        cap = cv2.VideoCapture(self.video_channel)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)        
        out = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (1280,720))

        while True:
            ret, frame = cap.read()
            color = (255, 255, 255)
            
            if not (cap.isOpened and ret):
                break

            if self.roi is None:
                self.roi = cv2.selectROI('roi', frame)
                cv2.destroyWindow('roi')
            
            (class_ids, scores, bboxes) =  self.model.detect(frame)

            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                class_name = self.classes[class_id]
                
                if class_name == "person":
                    res = [self.check_intersection(np.array(box), np.array(self.roi)) for box in bboxes]
                
                    if any(res):
                        detected("HD", 1)
                        color = (0, 0, 255)
                        cv2.putText(frame, 'SAKPAN ANG BOANG', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (55, 22, 255), 3)
                    else:
                        detected("HD", 0)

                cv2.rectangle(frame,(self.roi[0],self.roi[1]), (self.roi[0]+self.roi[2],self.roi[1]+self.roi[3]), color, 2)

            out.write(frame)
            cv2.imshow("Human Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                detected("HD", "")
                break
            
        out.release()
        cap.release()
        cv2.destroyAllWindows()

class FaceRecognition:
    def __init__(self, video_channel = 0, encodings='encodings.pickle', output='output', 
        detection_method='hog'):
        self.encodings = encodings
        self.output = output
        self.video_channel = video_channel
        self.detection_method = detection_method
        self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.data = pickle.loads(open(self.encodings, "rb").read())

    def face_recognize(self):
        cap = cv2.VideoCapture(0)
        writer = None

        while True:
            ret, frame = cap.read()
            color = (0, 255, 0)

            if ret is False:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = frame.shape[1] / float(rgb.shape[1])

            boxes = face_recognition.face_locations(rgb,
                model=self.detection_method)

            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(self.data["encodings"], encoding)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    for i in matchedIdxs:
                        name = self.data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    name = max(counts, key=counts.get)

                names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                detected("FR", 1)
                if name == "Unknown":
                    detected("FR", 0)
                    color = (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom),
                    color, 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, 2)

            if writer is None and self.output is not None:
                writer = cv2.VideoWriter(self.output, self.fourcc, 20,
                    (frame.shape[1], frame.shape[0]), True)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Face Recognition", frame)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                detected("FR", "")
                break

        cap.release()
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
            
class EncodeFaces:
    def __init__(self, dataset='dataset', encodings='encodings.pickle', 
        detection_method='cnn'):
        self.dataset = dataset
        self.encodings = encodings
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

        except:
            print("[ERROR] Something happened...")
        else:
            print("[INFO] Serializing encodings...")
            data = {"encodings": knownEncodings, "names": knownNames}
            f = open(self.encodings, "wb")
            f.write(pickle.dumps(data))
            f.close()


s = EncodeFaces()
s.encode_faces()