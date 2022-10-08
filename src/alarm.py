import json
import time
import threading

def read_face(data):
    if data['face-recognized']['ready'] == False:
        print("[ERROR] Face Recognition not running...")
        return

    if data['face-recognized']['found'] == True:
        print("[INFO] Face detected, authorize...")
    elif data['face-recognized']['found'] == False:
        print("[INFO] Face detected, unauthorize...")
    else:
        print("[INFO] No face detected...")


def read_human(data):
    if data['human-detected']['ready'] == False:
        print("[ERROR] Human Detection not running...")
        return

    if data['human-detected']['found']:
        print("[INFO] Human detected")
    else:
        print("[INFO] No Human detected...")


while True:
    with open('detect.json') as f:
        data = json.load(f)
        
    threading.Thread(target=read_face(data)).start()
    threading.Thread(target=read_human(data)).start()
    print('\n')

    time.sleep(1)
    