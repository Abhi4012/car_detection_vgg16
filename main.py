import os
import cv2
import numpy as np
import imutils
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

def load_data(images_path, annots_path):
    rows = open(annots_path).read().strip().split('\n')
    data = []
    targets = []
    filenames = []

    for row in rows[1:]:
        row = row.split(",")
        (filename, startX, startY, endX, endY) = row
        imagePath = os.path.join(images_path, filename)
        
        # Error handling: Check if the image is loaded successfully
        try:
            image = cv2.imread(imagePath)
            if image is None:
                print(f"Error loading image: {imagePath}")
                continue
        except Exception as e:
            print(f"Error loading image: {imagePath}. Exception: {e}")
            continue
        
        (h, w) = image.shape[:2]
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        data.append(image)
        targets.append((startX, startY, endX, endY))
        filenames.append(filename)

    data = np.array(data, dtype='float32') / 255.0
    targets = np.array(targets, dtype='float32')
    return data, targets, filenames

def create_model():
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
    vgg.trainable = False
    flatten = vgg.output
    flatten = Flatten()(flatten)
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    model = Model(inputs=vgg.input, outputs=bboxHead)
    return model

def train_model(model, train_images, train_targets, test_images, test_targets, num_epochs=25, batch_size=32, plot_path=None):
    opt = Adam(learning_rate=1e-4)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    H = model.fit(train_images, train_targets, validation_data=(test_images, test_targets), shuffle=True, batch_size=batch_size, epochs=num_epochs, verbose=1)
    if plot_path:
        plot_training_history(H, plot_path)
    return model, H

def plot_training_history(history, plot_path):
    N = len(history.history['loss'])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.title("Bounding Box Regression Loss on Training Set")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)

def save_model(model, model_path):
    model.save(model_path, save_format="h5")

def load_trained_model(model_path):
    return load_model(model_path)

def predict_bounding_box(model, image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)[0]
    return preds

def draw_bounding_box(image_path, preds):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    startX = int(preds[0] * w)
    startY = int(preds[1] * h)
    endX = int(preds[2] * w)
    endY = int(preds[3] * h)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()

def main():
    IMAGES_PATH = 'training_images'
    ANNOTS_PATH = 'train_solution_bounding_boxes (1).csv'
    BASE_OUTPUT = 'output'
    MODEL_PATH = os.path.join(BASE_OUTPUT, 'detector.h5')
    PLOT_PATH = os.path.join(BASE_OUTPUT, 'plot.png')
    TEST_FILENAMES = os.path.join(BASE_OUTPUT, 'test_images.txt')

    os.makedirs(BASE_OUTPUT, exist_ok=True)

    data, targets, filenames = load_data(IMAGES_PATH, ANNOTS_PATH)
    train_images, test_images, train_targets, test_targets, train_filenames, test_filenames = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)
    with open(TEST_FILENAMES, 'w') as f:
        f.write('\n'.join(test_filenames))

    model = create_model()
    trained_model, history = train_model(model, train_images, train_targets, test_images, test_targets, plot_path=PLOT_PATH)
    save_model(trained_model, MODEL_PATH)

    loaded_model = load_trained_model(MODEL_PATH)
    image_path = "./data/testing_images/vid_5_29460.jpg"
    preds = predict_bounding_box(loaded_model, image_path)
    draw_bounding_box(image_path, preds)

if __name__ == "__main__":
    main()