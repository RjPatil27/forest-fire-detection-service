import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import Adam
import cv2
from imutils import paths
import numpy as np
from PIL import Image as pil_image
from io import BytesIO

# Required Directories
TRAINING_DIR = os.path.sep.join(["image-data", "datasets", "training"])
VALIDATION_DIR = os.path.sep.join(["image-data", "datasets", "validation"])
MODEL_PATH = os.path.sep.join(["image-data", "model"])
IMAGE_PATH = os.path.sep.join(["image-data", "images"])
OUTPUT_PATH = os.path.sep.join(["image-data", "output"])

# Image Data Generator
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

# Customized CNN Models
def generate_customized_cnn_model(train_generator, validation_generator):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(96, (11, 11), strides=(
            4, 4), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(384, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['acc'])

    model.fit(
        train_generator,
        steps_per_epoch=15,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=15
    )

    model.save(os.path.sep.join([MODEL_PATH, "cnn_model.h5"]))
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

    # img = image.load_img(img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)/255

    # model = tf.keras.models.load_model(MODEL_PATH +'cnn_model.h5')
    classes = model.predict(x)
    print(np.argmax(classes[0]) == 0, max(classes[0]))

    if((np.argmax(classes[0]) == 0) == True):
        return cv2.putText(decoded, "Fire!!", (35, 35), cv2.FONT_HERSHEY_COMPLEX, 1.25, (0, 255, 0), 5)
    else:
        return cv2.putText(decoded, "No-Fire!!", (35, 35), cv2.FONT_HERSHEY_COMPLEX, 1.25, (0, 255, 0), 5)


def run_fire_and_smoke_detection_on_image(img):
    # generators = run_image_data_generator()
    # cnn_model = generate_customized_cnn_model(generators[0], generators[1])
    cnn_model = tf.keras.models.load_model(os.path.sep.join([MODEL_PATH, "cnn_model.h5"]))
    output_img = run_customized_cnn_model(img, cnn_model)
    return output_img
