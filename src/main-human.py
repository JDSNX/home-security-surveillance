import sys

from human.detection import HumanDetection

if __name__ == '__main__':
    args = sys.argv[1]

    if args == "run":
        s = HumanDetection('sample/video/video-1.mp4')
        s.detection()

    