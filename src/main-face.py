import sys

if __name__ == '__main__':
    args = sys.argv[1]

    if args == "train":
        from face.encoder import EncodeFaces
        from modules import get_images, create_env
        
        create_env()
        get_images()
        ef = EncodeFaces()
        ef.encode_faces()
    
    elif args == "run":
        from face.recognizer import FaceRecognition
        
        fr = FaceRecognition()
        fr.face_recognize()
    
    else:
        print('[ERROR] Wrong paramater passed...')