#!/usr/bin/env python3.6
# license removed for brevity
import pycuda.autoinit  # This is needed for initializing CUDA driver

import rospy
import rospkg

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolov4_trt_ros.msg import BoundingBox
from yolov4_trt_ros.msg import BoundingBoxes

import queue
import numpy

"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
#import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

WINDOW_NAME = 'TrtYOLO'

img_queue = queue.Queue()


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args

def main():
    rospack = rospkg.RosPack()
    pkgPath = rospack.get_path('yolov4_trt_ros')

    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile(pkgPath + '/weights/%s.trt' % args.model):
        raise SystemExit('ERROR: file (/weights/%s.trt) not found!' % args.model)

    #cam = Camera(args)
    #if not cam.isOpened():
    #    raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        640, 480)
    loop_and_detect(trt_yolo, args.conf_thresh, vis=vis)

    #cam.release()
    cv2.destroyAllWindows()


def loop_and_detect(trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        # img = cam.read()
        # if img is None:
        #     break
        if not img_q.empty():
        #print(type(img))
        #img = numpy.zeros((480,640,3), dtype=numpy.uint8)
            img_msg = img_q.get()
            img = numpy.frombuffer(img_msg.data, dtype=numpy.uint8).reshape(img_msg.height, img_msg.width, -1)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
            #print(len(boxes))
            #print(confs)
            bBoxes = BoundingBoxes()
            bBoxes.image_header = img_msg.header
            for i in range(len(boxes)):
                bBox = BoundingBox()
                bBox.xmin = boxes[i][0]
                bBox.ymin = boxes[i][1]
                bBox.xmax = boxes[i][2]
                bBox.ymax = boxes[i][3]
                bBox.Class = "robot"
                bBox.id = 0
                bBox.probability = confs[i]
                bBoxes.bounding_boxes.append(bBox)
                print("robot detected")
                print("robot ", i+1)
                print("probability : ", confs[i])
                
            boxPub.publish(bBoxes)
                        
            img = vis.draw_bboxes(img, boxes, confs, clss)
            img = show_fps(img, fps)
            print("FPS : ",fps)
            print("\n")
            image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
            pub.publish(image_message)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break
            elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
                full_scrn = not full_scrn
                set_display(WINDOW_NAME, full_scrn)
        else:
            
            print("wait images")

def imgCallback(img_msg):
    img_q.put(img_msg)

if __name__ == '__main__':
    img_q = queue.Queue()
    bridge = CvBridge()
    
    sub = rospy.Subscriber("/camera/infra1/image", Image, imgCallback)
    boxPub = rospy.Publisher('/bounding_boxes',BoundingBoxes, queue_size=10)
    pub = rospy.Publisher('detect_image', Image, queue_size=10)
    rospy.init_node('trt_yolo', anonymous=True)

    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

