import json
import os
from typing import Optional
import urllib.request as rq
from pymongo import MongoClient
from dotenv import load_dotenv

YOLO_WEIGHTS = 'dnn_model/yolov4-tiny.weights'
YOLO_CFG = 'dnn_model/yolov4-tiny.cfg'
load_dotenv()

client = MongoClient(os.getenv('mongodb'))
db = client.get_database(os.getenv('db'))
print("[INFO] Connecting database...")

def get_classes():
    return [class_name.strip() for class_name in open("dnn_model/classes.txt").readlines()]


def is_ready(type, ready):
    collection = db.detect

    print(f"[INFO] {type} ready - {ready}...")

    if type == "face-recognized":
        collection.update_one(
            { 'tech': 'face_recognition' },
            { '$set': {'is_ready': ready},
            '$currentDate': { 'last_modified': True}}
        )

    elif type == "human-detected":
        collection.update_one(
            { 'tech': 'human_detection' },
            { '$set': {'is_ready': ready},
            '$currentDate': { 'last_modified': True}}
        )


def detected(type, is_detected, name: Optional[str]=None) -> None:
    collection = db.detect

    print(f"[INFO] {type} detected - {is_detected}...")

    if type == "face-recognized":
        collection.update_one(
            { 'tech': 'face_recognition' },
            { '$set': {
                'is_detected': is_detected,
                'last_recognized': name
            },
            '$currentDate': { 'last_modified': True}}
        )

    elif type == "human-detected":
        collection.update_one(
            { 'tech': 'human_detection' },
            { '$set': {'is_detected': is_detected},
            '$currentDate': { 'last_modified': True}}
        )


def get_images():
    try:
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