import json
import os
from typing import Optional
import urllib.request as rq
from pymongo import MongoClient
from dotenv import load_dotenv

YOLO_WEIGHTS = 'dnn_model/yolov4-tiny.weights'
YOLO_CFG = 'dnn_model/yolov4-tiny.cfg'
load_dotenv()


def get_classes():
    return [class_name.strip() for class_name in open("dnn_model/classes.txt").readlines()]


def is_ready(type, ready):
    with open('detect.json') as f:
        data = json.load(f)

    data[type]['ready'] = ready

    with open('detect.json', "w") as f:
        json.dump(data, f)


def detected(type, is_detected) -> None:
    with open('detect.json') as f:
        data = json.load(f)

    data[type]['found'] = is_detected

    with open('detect.json', "w") as f:
        json.dump(data, f)


def get_images():
    try:
        client = MongoClient(os.getenv('mongodb'))
        db = client.get_database(os.getenv('db'))
        print("[INFO] Connecting database...")
        images = db.tests
        images.count_documents({})
        images = list(images.find())

        if os.path.isdir('dataset'):
            for _, v in enumerate(images):
                for name, image in v['Photos'].items():
                    if not os.path.exists(f'dataset/{name}'):
                        os.mkdir(f'dataset/{name}')

                    for i, iu in enumerate(image):
                        image_url = iu['image_url'].split('images/')[1]
                        file = os.path.join(os.getcwd(), f'dataset/{name}', image_url)
                        rq.urlretrieve(iu['image_url'], file)
                        print(f"[INFO] Retrieving {name} images {i+1}/{len(image)}...")
    except:
        print(f"[ERROR] Something happened on getting image...")
    else:
        print(f"[INFO] Dataset complete...")

get_images()