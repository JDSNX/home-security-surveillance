YOLO_WEIGHTS = 'dnn_model/yolov4-tiny.weights'
YOLO_CFG = 'dnn_model/yolov4-tiny.cfg'

def get_classes():
    return [class_name.strip() for class_name in open("dnn_model/classes.txt").readlines()]