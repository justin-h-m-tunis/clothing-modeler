from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2
import serial
import scipy.optimize
from Motor import *
from Camera import *

if __name__ == '__main__':
    motor = Motor(macrostep_time=160,total_macrosteps=200,baudrate=9600,com='COM3',onSerialFail=lambda : print("Cannot find motor!"))
    camera = Camera()
    motor.fullRotation(cond=lambda num: camera.captureRGBD(num, show_image=False))

