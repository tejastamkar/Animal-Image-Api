import numpy as np
import pandas as pd
import os
import cv2
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests

# for line in urllib2.urlopen(target_url):
#     print line

# weights_path = urlopen(
#     "https://firebasestorage.googleapis.com/v0/b/test-8ecf6.appspot.com/o/yolov3.weights?alt=media&token=bab04113-4f7b-4d1d-a178-b4483440954a")

response = requests.get("https://firebasestorage.googleapis.com/v0/b/test-8ecf6.appspot.com/o/yolov3.weights?alt=media&token=bab04113-4f7b-4d1d-a178-b4483440954a")
open("yolov3.weights", "wb").write(response.content)

weights_path = './yolov3.weights';
configuration_path = './model/yolov3.cfg'
labels = open('./model/coco.names').read().strip().split('\n')
probability_minimum = 0.5
threshold = 0.3
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
layersnamesall = network.getLayerNames()
layers_names_output = [layersnamesall[i-1]
                       for i in network.getUnconnectedOutLayers()]


def ImagePath(path):
    bounding_boxes = []
    confidences = []
    class_numbers = []
    image_input = cv2.imread(path)
    blob = cv2.dnn.blobFromImage(
        image_input, 1/255.0, (416, 416), swapRB=True, crop=False)
    # blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)
    h, w = image_input.shape[:2]
    for result in output_from_network:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype(
                    'int')
                x_min = int(x_center-(box_width/2))
                y_min = int(y_center-(box_height/2))
                bounding_boxes.append(
                    [x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

        for item in sorted(set(class_numbers)):
            OutputDec = labels[item]

        return OutputDec

