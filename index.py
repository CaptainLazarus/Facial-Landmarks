import numpy as np
import cv2
import imutils
import argparse
import dlib
from imutils import face_utils

def initArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i" , "--image" , required=True)
    parser.add_argument("-p" , "--shape" , required=True)
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = initArgs()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape"])

    orig = cv2.imread(args["image"])
    image = orig.copy()
    image = imutils.resize(image , width=600)
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    rects = detector(gray , 1)

    for (i,rect) in enumerate(rects):
        shape = predictor(gray , rect)
        shape = face_utils.shape_to_np(shape)

        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image , (x,y) , (x+w , y+h) , (0,255,0) , 2)

        cv2.putText(image , "Face #{}".format(i+1) , (x-10,y-10) , cv2.FONT_HERSHEY_SIMPLEX , 0.6 , (0,255,255) , 2)

        for (x,y) in shape:
            cv2.circle(image , (x,y) , 1 , (0,0,255) , -1)

    cv2.imshow("Image" , image)
    cv2.waitKey(0)