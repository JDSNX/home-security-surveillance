import sys

from hdfr_class import EncodeFaces, FaceRecognition

if __name__ == '__main__':
    args = sys.argv[1]

    if args == "train":
        ef = EncodeFaces()
        ef.encode_faces()
    
    elif args == "run":
        fr = FaceRecognition()
        fr.face_recognize()

    