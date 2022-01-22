import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)'
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model

import os
import cv2
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
RESNET50_DIR = os.path.sep.join(["video-data", "resnet50"])
VGG16_DIR = os.path.sep.join(["video-data", "vgg16"])
IMAGE_PATH = os.path.sep.join(["video-data", "img_data"])
MODEL_PATH = os.path.sep.join(["video-data", "model"])
VIDEO_PATH = os.path.sep.join(["video"])
label_dict_l = {0: 'default', 1: 'fire', 2: 'smoke'}

IMG_SIZE = 224
NUM_EPOCHS = 10
NUM_CLASSES = 3
TRAIN_BATCH_SIZE = 77
TEST_BATCH_SIZE = 1


def draw_prediction(frame, class_string):
    print('draw_prediction')
    x_start = frame.shape[1] - 600
    cv2.putText(frame, class_string, (x_start, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 2, cv2.LINE_AA)
    return frame


def prepare_image_for_prediction(img):
    print('prepare_image_for_prediction')
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    # The below function inserts an additional dimension at the axis position provided
    img = np.expand_dims(img, axis=0)
    # perform pre-processing that was done when resnet model was trained.
    return preprocess_input(img)


def get_display_string(pred_class, label_dict):
    print('get_display_string')
    txt = ""
    for c, confidence in pred_class:
        txt += label_dict[c]
        if c:
            txt += '[' + str(confidence) + ']'
    #print("count="+str(len(pred_class)) + " txt:" + txt)
    return txt


def predict(model, video_path, filename, label_dict):
    print('predict')
    vs = cv2.VideoCapture(video_path)
    fps = math.floor(vs.get(cv2.CAP_PROP_FPS))
    ret_val = True
    writer = 0

    while True:
        ret_val, frame = vs.read()
        if not ret_val:
            break

        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_for_pred = prepare_image_for_prediction(resized_frame)
        pred_vec = model.predict(frame_for_pred)
        # print(pred_vec)
        pred_class = []
        confidence = np.round(pred_vec.max(), 2)

        if confidence > 0.4:
            pc = pred_vec.argmax()
            pred_class.append((pc, confidence))
        else:
            pred_class.append((0, 0))
        if pred_class:
            txt = get_display_string(pred_class, label_dict)
            frame = draw_prediction(frame, txt)

        if not writer:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(
                filename, fourcc, fps, (frame.shape[1], frame.shape[0]), True)

        # write the out
        writer.write(frame)

    vs.release()
    writer.release()


# model = create_model('S')
# trained_model_s, train_generator, validation_generator = train_model(model)
# label_dict_s = get_label_dict(train_generator)

def run_fire_and_smoke_detection_on_video(file_name):
    model = load_model(os.path.sep.join([MODEL_PATH, "resnet_cnn.h5"]))
    video_path = os.path.sep.join([VIDEO_PATH, file_name])
    result_path = os.path.sep.join([VIDEO_PATH, 'result.avi'])
    predict(model, video_path, result_path,  label_dict_l)
    return result_path
    
