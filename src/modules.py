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
        images = db.images
        images.count_documents({})
        images = list(images.find())

        if os.path.isdir('dataset'):
            for i, image in enumerate(images):
                name = image['name']
                image_url = image['image_url'].split('images/')[1]
                if not os.path.exists(f'dataset/{name}'):
                    os.mkdir(f'dataset/{name}')

                file = os.path.join(os.getcwd(), f'dataset/{name}', image_url)
                rq.urlretrieve(image['image_url'], file)            
                print(f"[INFO] Retrieving images {i+1}/{len(images)}...")

    except:
        print(f"[ERROR] Something happened on getting image...")
    else:
        print(f"[INFO] Dataset complete...")