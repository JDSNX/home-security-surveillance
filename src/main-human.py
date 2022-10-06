import sys

from human.detection import HumanDetection

if __name__ == '__main__':
    args = sys.argv[1]

    if args == "run":
        s = HumanDetection('video.mp4')
        s.detection()

    