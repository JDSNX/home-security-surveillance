import cv2
import numpy as np

from datetime import datetime

from modules import YOLO_CFG, YOLO_WEIGHTS, get_classes

class HumanDetection:
    def __init__(self, video_channel=None, roi=None, output_name=None):
        self.net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(320, 320), scale=1/255)

        self.width = int(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = int(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = int(cv2.CAP_PROP_FPS)
        self.output_name = f'{datetime.utcnow}_output.avi' if output_name is None else output_name
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
        out = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (self.width, self.height))

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
                        color = (0, 0, 255)
                        cv2.putText(frame, 'SAKPAN ANG BOANG', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (55, 22, 255), 3)
    
                cv2.rectangle(frame,(self.roi[0],self.roi[1]), (self.roi[0]+self.roi[2],self.roi[1]+self.roi[3]), color, 2)

            out.write(frame)
            cv2.imshow("Human Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    s = HumanDetection('video.mp4')
    s.detection()

    