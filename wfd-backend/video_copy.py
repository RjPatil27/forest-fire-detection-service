import numpy as np  # linear algebra
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.keras import optimizers

import cv2
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
RESNET50_DIR = os.path.sep.join(["video-data", "resnet50"])
VGG16_DIR = os.path.sep.join(["video-data", "vgg16"])
IMAGE_PATH = os.path.sep.join(["video-data", "img_data"])
VIDEO_PATH = os.path.sep.join(["video-data", "video_data"])
VIDEO_OUTPUT = os.path.sep.join(["video-output"])
MODEL_PATH = os.path.sep.join(["video-data", "model"])

IMG_SIZE = 224
NUM_EPOCHS = 20
NUM_CLASSES = 3
TRAIN_BATCH_SIZE = 77
TEST_BATCH_SIZE = 1


def create_model(model_size):
    print('create_model')
    model = Sequential()
    if model_size == 'L':
        resnet_weights_path = os.path.sep.join([RESNET50_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"])
        resnet = ResNet50(include_top=False, pooling='avg',
                          weights='imagenet')
        # resnet.summary()
        model.add(resnet)
        model.layers[0].trainable = False
    else:
        vgg_weights_path = os.path.sep.join([VGG16_DIR, "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"])
        vgg = VGG16(include_top=False, weights=vgg_weights_path)
        vgg.summary()
        model.add(vgg)
        model.add(GlobalAveragePooling2D())
        model.layers[0].trainable = False
        model.layers[1].trainable = False

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Say no to train first layer (ResNet) model. It is already trained

    opt = optimizers.Adam()
    model.compile(
        optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model):
    print('train_model')
    #ata_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                                 width_shift_range=0.1,
                                                 height_shift_range=0.1,
                                                 # sear_range=0.01,
                                                 zoom_range=[0.9, 1.25],
                                                 horizontal_flip=True,
                                                 vertical_flip=False,
                                                 data_format='channels_last',
                                                 brightness_range=[0.5, 1.5]
                                                 )

    train_generator = data_generator_with_aug.flow_from_directory(
        os.path.sep.join([IMAGE_PATH, "train"]),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical')

    validation_generator = data_generator_with_aug.flow_from_directory(
        os.path.sep.join([IMAGE_PATH, "test"]),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        class_mode='categorical')

    #y_train = get_labels(train_generator)
    #weights = class_weight.compute_class_weight('balanced',np.unique(y_train), y_train)
    #dict_weights = { i: weights[i] for i in range(len(weights)) }

    H = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n/TRAIN_BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=1  # ,
        # class_weight=dict_weights
    )

    # plot_history(H, NUM_EPOCHS)

    return model, train_generator, validation_generator


def get_label_dict(train_generator):
    print('get_label_dict')
    # Get label to class_id mapping
    labels = (train_generator.class_indices)
    label_dict = dict((v, k) for k, v in labels.items())
    return label_dict


def get_labels(generator):
    print('get_labels')
    generator.reset()
    labels = []
    for i in range(len(generator)):
        labels.extend(np.array(generator[i][1]))
    return np.argmax(labels, axis=1)


# def get_pred_labels(test_generator):
#     print('get_pred_labels')
#     test_generator.reset()
#     pred_vec = model.predict_generator(test_generator,
#                                        steps=test_generator.n,  # test_generator.batch_size
#                                        verbose=1)
#     return np.argmax(pred_vec, axis=1), np.max(pred_vec, axis=1)


def plot_history(H, NUM_EPOCHS):
    print('plot_history')
    plt.style.use("ggplot")
    fig = plt.figure()
    fig.set_size_inches(15, 5)

    fig.add_subplot(1, 3, 1)
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Validation Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    fig.add_subplot(1, 3, 2)
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["acc"], label="train_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    fig.add_subplot(1, 3, 3)
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_acc"], label="val_acc")
    plt.title("Validation Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    plt.show()


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
        # print(pred_class)
        # plt.axis('off')
        # plt.imshow(frame)
        # plt.show()
        #clear_output(wait = True)
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

def create_model_save():

    model = create_model('L')
    trained_model_l, train_generator, validation_generator = train_model(model)
    trained_model_l.save(os.path.sep.join([MODEL_PATH, "resnet_cnn.h5"]))
    # label_dict_l = get_label_dict(train_generator)
    # print(label_dict_l)
    # plot_history(trained_model_l,NUM_EPOCHS)


# video_path = os.path.sep.join([VIDEO_PATH, 'test_videos', 'test1.mp4'])
# predict(trained_model_l, video_path, os.path.sep.join([VIDEO_OUTPUT, 'test1_9.avi']),  label_dict_l)
#
# video_path = os.path.sep.join([VIDEO_PATH, 'test_videos', 'test2.mp4'])
# predict(trained_model_l, video_path, os.path.sep.join([VIDEO_OUTPUT, 'test2_9.avi']),  label_dict_l)
#
# video_path = os.path.sep.join([VIDEO_PATH, 'test_videos', 'test3.mp4'])
# predict(trained_model_l, video_path, os.path.sep.join([VIDEO_OUTPUT, 'test3_9.avi']),  label_dict_l)
