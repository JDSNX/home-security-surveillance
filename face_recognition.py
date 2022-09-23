import os
import cv2
import numpy as np
import pickle
import pandas as pd
import urllib
import string
import random
import sys

from pymongo import MongoClient
from PIL import Image
from datetime import datetime

class FaceRecognition:
    def __init__(self, face_cascade=None, output_file=None):
        self.face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml') if face_cascade is None else cv2.CascadeClassifier(face_cascade)
        self.output_file = f'{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}_video.avi' if output_file is None else output_file
        self.client = MongoClient("mongodb+srv://jyuviolegrace:androssi@homesecurity.bzuij.mongodb.net/ReactNativeApp?retryWrites=true&w=majority") 

    def train(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "images")
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    id_ = label_ids[label]
                    pil_image = Image.open(path).convert("L")
                    image_array = np.array(pil_image, "uint8")
                    faces = self.face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

                    for (x, y, w, h) in faces:
                        roi = image_array[y:y+h, x:x+h]
                        x_train.append(roi)
                        y_labels.append(id_)

        with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trainner.yml")

    def recognition(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainner.yml")

        labels = {"person_name": 1}
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v:k for k, v in og_labels.items()}

        cap = cv2.VideoCapture(0) ##c0 - laptop's camera; 1 - external cam
        wwidth, hheight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'XVID'), 20, (wwidth, hheight))

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
                
            for(x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                id_, conf = self.recognizer.predict(roi_gray)
                if conf >= 45 and conf <= 85:
                    print(id_)
                    print(labels[id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    det = cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

                rec = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            writer.write(frame)
            cv2.imshow('Face Recognition', frame)     

            if cv2.waitKey(1) & 0xFF == 27:
                break
            
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
    
    def fetch(self):
        db = self.client.get_database('ReactNativeApp')
        records = db.images
        records.count_documents({}) #counting of number of documents
        all_rec = records.find()
        list_rec = list(all_rec)
        df = pd.DataFrame(list_rec)
        
        path = "D:/Files/Thesis 2/Project/Project VS Code/env/images"

        for nname in df.name:
            if len(os.listdir(path)) == 0:
                os.mkdir(os.path.join(path, nname))
                print("done")
            else:
                for root, subdirs, files in os.walk(path):
                    for d in subdirs:
                        if d == nname:
                            print("Folder Name " + nname + " Already Exist")
                            break
                        elif d != nname:
                            print(nname)
                            try:
                                os.mkdir(os.path.join(path, nname))
                                print("done")
                            except:
                                print("error")
    
        for namee in df.name:
            #img_url = df.loc[(df['name'] == namee), 'image_url']
            for img_u in df.loc[(df['name'] == namee), 'image_url']:
                letters = string.ascii_lowercase
                file_n = ''.join(random.choice(letters) for i in range(5))
                urllib.request.urlretrieve(img_u, 'D:/Files/Thesis 2/Project/Project VS Code/env/images/'+ namee + '/' + file_n + '.jpg')

if __name__ == "__main__":
    args = sys.argv[1]

    fr = FaceRecognition()
    
    if args == 'run':
        fr.recognition()

    elif args == 'train':
        fr.fetch()
        fr.train()