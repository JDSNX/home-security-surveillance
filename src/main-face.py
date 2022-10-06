import sys

if __name__ == '__main__':
    args = sys.argv[1]

    if args == "train":
        from face.encoder import EncodeFaces
        from modules import get_images

        get_images()
        ef = EncodeFaces()
        ef.encode_faces()
    
    elif args == "run":
        from face.recognizer import FaceRecognition
        
        fr = FaceRecognition()
        fr.face_recognize()

    