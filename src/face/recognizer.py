import face_recognition
import pickle
import cv2

from modules import detected

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


    def face_recognize(self):
        cap = cv2.VideoCapture(self.video_channel)
        writer = None

        while True:
            ret, frame = cap.read()
            color = (0, 255, 0)

            if ret is False:
                print('[ERROR] Something wrong with your camera...')
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = frame.shape[1] / float(rgb.shape[1])

            boxes = face_recognition.face_locations(rgb, model=self.detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            names = []
            if boxes and encodings:
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

        if writer is not None:
            writer.release()
            
        cap.release()
        cv2.destroyAllWindows()