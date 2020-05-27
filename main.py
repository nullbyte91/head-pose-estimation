import logging as log 
import cv2 
from imutils.video import FPS
import numpy as np
from collections import namedtuple

from argparse import ArgumentParser

from openvino.inference_engine import IENetwork, IECore
from src import faceDetector, headPos_Estimator
from math import cos, sin, pi

CPU_DEVICE_NAME = "CPU"

FaceInferenceResults = namedtuple('Point', 'x y')

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-m_fd", "--mode_face_detection", required=True, type=str,
                        help="Path to an .xml file with a trained Face Detection model")
    parser.add_argument("-m_hp", "--model_head_position", required=True, type=str,
                        help="Path to an .xml file with a trained Head Pose Estimation model")                                 
    return parser

def main():
    
    # Set log to INFO
    log.basicConfig(level=log.INFO)

    # Grab command line args
    args = build_argparser().parse_args()

    # Handle the input stream
    try:
        cap = cv2.VideoCapture(args.input)
    except FileNotFoundError:
        print("Cannot locate video file: "+ args.input)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    # Initialize the plugin
    ie = IECore()


    # Initialize the plugin
    ie = IECore()

    # Face Detection init
    face_detection = faceDetector.FaceDetector()
    face_detection.load_model(ie, args.mode_face_detection, "CPU", num_requests=0)

    # Head Position init
    head_position  = headPos_Estimator.HeadPosEstimator()
    head_position.load_model(ie, args.model_head_position, "CPU", num_requests=0)

    # Get a Input blob shape of face detection
    _, _, in_h_f, in_w_f = face_detection.get_input_shape()

    fps = FPS().start()
    
    while cap.isOpened():
        #Read the next frame
        _, frame = cap.read()
        if frame is None:
            break

        fh = frame.shape[0]
        fw = frame.shape[1]
        
        key_pressed = cv2.waitKey(50)
    
        image_resize = cv2.resize(frame, (in_w_f, in_h_f), interpolation = cv2.INTER_AREA)
        image = np.moveaxis(image_resize, -1, 0)

        # Perform inference on the frame
        face_detection.exec_net(image, request_id=0)
        
        headPoseAngles = {
            "p": 0,
            "r": 0,
            "y": 0
        }
        # Get the output of inference
        if face_detection.wait(request_id=0) == 0:
            # Get Face detection
            detection = face_detection.get_output(request_id=0)
            for i in range(0, detection.shape[2]):
                confidence = detection[0, 0, i, 2]
                # If confidence > 0.5, save it as a separate file
                if (confidence > 0.5):
                    xmin = int(detection[0, 0, i, 3] * fw)
                    ymin = int(detection[0, 0, i, 4] * fh)
                    xmax = int(detection[0, 0, i, 5] * fw)
                    ymax = int(detection[0, 0, i, 6] * fh)
                    xmax = max(1, min(xmax, fw - 1))
                    ymax = max(1, min(ymax, fh - 1))
                    xmin = max(0, min(xmin, xmax - 1))
                    ymin = max(0, min(ymin, ymax - 1))

                    # Head position
                    image_fc = frame[ymin:ymax+1, xmin:xmax+1]
                    # Get a Input blob shape of head position
                    in_h_n, in_h_c, in_h_h, in_h_w = head_position.get_input_shape()
                    image_h = cv2.resize(image_fc, (in_h_w, in_h_h), interpolation = cv2.INTER_AREA)
                    image_h = np.moveaxis(image_h, -1, 0)
                    
                    head_position.exec_net(image_h, request_id=0)
                    if head_position.wait(request_id=0) == 0:
                        head_positions = head_position.get_output(request_id=0)
                        headPoseAngles['y'] = head_positions["angle_y_fc"][0]
                        headPoseAngles['p'] = head_positions["angle_p_fc"][0]
                        headPoseAngles['r'] = head_positions["angle_r_fc"][0]
                        cos_r = cos(headPoseAngles['r'] * pi / 180)
                        sin_r = sin(headPoseAngles['r'] * pi / 180)
                        sin_y = sin(headPoseAngles['y'] * pi / 180)
                        cos_y = cos(headPoseAngles['y'] * pi / 180)
                        sin_p = sin(headPoseAngles['p'] * pi / 180)
                        cos_p = cos(headPoseAngles['p'] * pi / 180)

                        x = int((xmin + xmax) / 2)
                        y = int((ymin + ymax) / 2)

                        # center to right
                        cv2.line(frame, (x,y), (x+int(50*(cos_r*cos_y+sin_y*sin_p*sin_r)), y+int(50*cos_p*sin_r)), (0, 0, 255), thickness=3)
                        # center to top
                        cv2.line(frame, (x, y), (x+int(50*(cos_r*sin_y*sin_p+cos_y*sin_r)), y-int(50*cos_p*cos_r)), (0, 255, 0), thickness=3)
                        # center to forward
                        cv2.line(frame, (x, y), (x + int(50*sin_y*cos_p), y + int(50*sin_p)), (255, 0, 0), thickness=3)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        #Break if escape key pressed
        if key_pressed == 27:
            break

if __name__ == "__main__":
    main()