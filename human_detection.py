import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules import YOLO_CFG, YOLO_WEIGHTS, get_classes

class HumanDetection:
    def __init__(self, video_channel=None, roi=None):
        self.net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(320, 320), scale=1/255)

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
                # (x, y, w, h) = bbox
                class_name = self.classes[class_id]

                pad_w, pad_h = 0, 0#int(0.15*w), int(0.05*h)
                
                if class_name == "person":
                    res = [self.check_intersection(np.array(box), np.array(self.roi)) for box in bboxes]
                
                    if any(res):
                        color = (0, 0, 255)
                        cv2.putText(frame, 'SAKPAN ANG BOANG', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (55, 22, 255), 3)
    
                cv2.rectangle(frame,(self.roi[0],self.roi[1]), (self.roi[0]+self.roi[2],self.roi[1]+self.roi[3]), color, 2)

                    # cv2.putText(frame, f'{class_name}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 22, 255), 1)
                    # cv2.rectangle(frame, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), color, 1)

            cv2.imshow("HumanDetection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    s = HumanDetection('video.mp4')
    s.detection()

    