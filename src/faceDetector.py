#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

import argparse
import cv2
import numpy as np

class FaceDetector:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None 
        self.exec_network = None
        self.infer_request = None


    def load_model(self, ie, model, device="CPU", num_requests=0):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Initialize the plugin
        self.plugin = IECore()

        # Read the IR as a IENetwork
        try:
            self.network = IENetwork(model=model_xml, weights=model_bin)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device, num_requests=num_requests)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape
    
    def get_output_name(self):
        '''
        Gets the input shape of the network
        '''
        output_name, _ = "", self.network.outputs[next(iter(self.network.outputs.keys()))]
        for output_key in self.network.outputs:
            if self.network.layers[output_key].type == "DetectionOutput":
                output_name, _ = output_key, self.network.outputs[output_key]
        
        if output_name == "":
            log.error("Can't find a DetectionOutput layer in the topology")
            exit(-1)
        return output_name

    def exec_net(self, image, request_id):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=request_id, 
            inputs={self.input_blob: image})
        return

    
    def wait(self, request_id):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[request_id].wait(-1)
        return status


    def get_output(self, request_id):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[request_id].outputs[self.output_blob]


def main(args):
    ie = IECore()
    faceDetector = FaceDetector()
    faceDetector.load_model(ie, args.model, "CPU", num_requests=0)
    # Get a Input blob shape of face detection
    _, _, in_h, in_w = faceDetector.get_input_shape()
    
    image_o = cv2.imread(args.input)
    fh = image_o.shape[0]
    fw = image_o.shape[1]
    
    image_resize = cv2.resize(image_o, (in_w, in_h), interpolation = cv2.INTER_AREA)
    image = np.moveaxis(image_resize, -1, 0)

    # Perform inference on the frame
    faceDetector.exec_net(image, request_id=0)

    # Get the output of inference
    if faceDetector.wait(request_id=0) == 0:
        detection = faceDetector.get_output(request_id=0)
        for i in range(0, detection.shape[2]):
                confidence = detection[0, 0, i, 2]
                # If confidence > 0.5, save it as a separate file
                if (confidence > 0.5):
                    faceBoundingBox = detection[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])
                    (startX, startY, endX, endY) = faceBoundingBox.astype("int")
                    # image_fc = frame[startY:endY, startX:endX]
                    cv2.rectangle(image_o, (startX, startY), (endX, endY), (0, 125, 255), 3)
                    image_fc = image_o[startY:endY, startX:endX]
                    cv2.imwrite('resource/facedetection.png', image_fc)

    cv2.imshow('frame', image_o)
    cv2.waitKey(0)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=False, type=str,
                    default="mo_model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml")
    parser.add_argument("-i", "--input", required=False, type=str, 
                        default='resource/example_02.jpg', help="Path to image or video file")
    args=parser.parse_args()
    main(args)