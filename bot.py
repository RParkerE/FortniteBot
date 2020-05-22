# Fortnite Bot

import cv2
import ctypes
import numpy as np
import directkeys
from PIL import Image, ImageGrab
from MobileNetSSD import MobileNetSSD
from matplotlib import cm

class FortniteBot(object):
    def __init__(self):
        self.prototxt = ".\\MobileNetSSD_deploy.prototxt"
        self.weights = ".\\MobileNetSSD_deploy.caffemodel"

        self.screenRecord()
    def screenRecord(self):
        while True:
            screen =  np.array(ImageGrab.grab(bbox=(0,0,1366,768)))
            new_screen = self.process_img(screen)
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", new_screen)
            #cv2.imshow('window', image = Image.fromarray(new_screen.astype('uint8'), 'RGB'))
            #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    def process_img(self, image):
        original_image = image
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ssd =  MobileNetSSD(original_image, self.prototxt, self.weights)
        detect_img = ssd.detect()
        return detect_img

f = FortniteBot()
