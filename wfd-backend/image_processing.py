
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import tensorflow as tf

from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import SGD
from skimage import measure
from PIL import Image as pil_image
from io import BytesIO


TRAINING_DIR = os.path.sep.join(["image-data", "datasets", "training"])
VALIDATION_DIR = os.path.sep.join(["image-data", "datasets", "validation"])
MODEL_PATH = os.path.sep.join(["image-data", "model"])
IMAGE_PATH = os.path.sep.join(["image-data", "test"])
OUTPUT_PATH = os.path.sep.join(["image-data", "output"])


def run_image_data_generator():
    training_datagen = ImageDataGenerator(rescale=1./255,
                                          horizontal_flip=True,
                                          rotation_range=30,
                                          height_shift_range=0.2,
                                          fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                           target_size=(
                                                               224, 224),
                                                           class_mode='categorical',
                                                           batch_size=64)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=16)
    return [train_generator, validation_generator]


def generate_customized_inceptionv3_cnn_model(train_generator, validation_generator):
    #
    # Customized CNN Model (Basic Model)
    input_tensor = Input(shape=(224, 224, 3))
    base_model = InceptionV3(input_tensor=input_tensor,
                             weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['acc'])

    history = model.fit(
        train_generator,
        steps_per_epoch=14,
        epochs=14,
        validation_data=validation_generator,
        validation_steps=14)

    for layer in model.layers[:249]:
        layer.trainable = False

    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['acc'])

    history = model.fit(
        train_generator,
        steps_per_epoch=14,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=14)

    # Graph representation of CNN model training and validation
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # epochs = range(len(acc))

    # plt.plot(epochs, acc, 'g', label='Training accuracy')
    # plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.legend(loc=0)
    # # plt.figure()
    # plt.savefig('TrainingandValidationAccuracy')
    # plt.show()

    # plt.plot(epochs, loss, 'r', label='Training loss')
    # plt.plot(epochs, val_loss, 'orange', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend(loc=0)
    # # plt.figure()
    # plt.savefig('TrainingandValidationLoss')
    # plt.show()

    # Save Basic Customized CNN model
    model.save(os.path.sep.join(
        [MODEL_PATH, "customized_inceptionv3_cnn_model.h5"]))
    return model


def run_customized_cnn_model(img, model):
    nparr = np.fromstring(img, np.uint8)
    # decode image
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # original = cv2.imread(decoded)
    # cv2.imshow("Test",decoded)
    # cv2.waitKey()

    img = pil_image.open(BytesIO(img))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    width_height_tuple = (224, 224)
    if img.size != width_height_tuple:
        img = img.resize(width_height_tuple, pil_image.NEAREST)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)/255

    classes = model.predict(x)
    print(np.argmax(classes[0]) == 0, max(classes[0]))

    if((np.argmax(classes[0]) == 0) == True):
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        #
        # labels = measure.label(thresh, neighbors=8, background=0)
        labels = measure.label(thresh, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")

        for label in np.unique(labels):
            if label == 0:
                continue

            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            if numPixels > 200:
                mask = cv2.add(mask, labelMask)

        cnts = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # cnts = contours.sort_contours(cnts)[2]

        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            # print(x,y,w,h)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(decoded, (int(cX), int(cY)), int(radius),
                       (0, 0, 255), 2)
            cv2.putText(decoded, "Fire/Smoke!!", (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1.25, (0, 0, 255), 3)
            # cv2.putText(img1, "#{}".format(i + 1), (x, y - 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # cv2.imshow("Output Image(Picture)", img)
        # cv2.waitKey(0)
        return decoded

    else:
        return cv2.putText(decoded, "Non-Fire!!", (50, 50),
                           cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        # cv2.imshow("Prediction",op)
        # filename = "{}.jpg".format(i)
        # # cv2.imwrite(os.path.sep.join([OUTPUT_PATH,filename]), op)
        # cv2.imshow("Non Fire Image", op)
        # cv2.waitKey()


def run_fire_and_smoke_detection_on_image(img):
    # generators = run_image_data_generator()
    # cnn_model = generate_customized_inceptionv3_cnn_model(generators[0], generators[1])
    cnn_model = tf.keras.models.load_model(os.path.sep.join(
        [MODEL_PATH, "customized_inceptionv3_cnn_model.h5"]))
    output_img = run_customized_cnn_model(img, cnn_model)
    return output_img
