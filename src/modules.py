import json
import os
import urllib.request as rq
import cv2
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Optional

YOLO_WEIGHTS = 'dnn_model/yolov4-tiny.weights'
YOLO_CFG = 'dnn_model/yolov4-tiny.cfg'
load_dotenv()

client = MongoClient(os.getenv('mongodb'))
db = client.get_database(os.getenv('db'))
print("[INFO] Connecting database...")

def get_classes():
    return [class_name.strip() for class_name in open("dnn_model/classes.txt").readlines()]


async def is_ready(type, ready):
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


async def detected(type, is_detected, name: Optional[str]=None) -> None:
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


async def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    
    # x1 = top
    # x2 = left
    # x3 = right
    # x4 = bottom

    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


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

def create_env():
    if not os.path.exists('.env'):
        print('[INFO] Creating .env file...')
        f = open('.env','a')
        f.write('mongodb=\n')
        f.write('db=ReactNativeApp')
        f.close()
        print('[INFO] .env file is created...')