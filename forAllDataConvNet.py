import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import load_img
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)


def load_dataset():
    """
        Load the CIFAR-10 dataset
    """

    DATASET_DIR = './generated_dataset_learn_smallset10'

    image_paths = []
    target_labels = []

    for filename in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR, filename)
        temp = filename.split('_')
        age = int(temp[0])
        image_paths.append(image_path)

        if age <= 2:
            target_labels.append(1)
        elif age <= 11:
            target_labels.append(2)
        elif age <= 21:
            target_labels.append(3)
        elif age <= 25:
            target_labels.append(4)
        elif age <= 27:
            target_labels.append(5)
        elif age <= 31:
            target_labels.append(6)
        elif age <= 38:
            target_labels.append(7)
        elif age <= 48:
            target_labels.append(8)
        elif age <= 58:
            target_labels.append(9)
        else:
            target_labels.append(0)

    df = pd.DataFrame()
    df['image'], df['target'] = image_paths, target_labels
    X = extract_features(df['image'])
    Y = df['target']

# learn data
    DATASET_DIR1 = './generated_dataset_validation_smallset10'

    image_paths1 = []
    target_labels1 = []

    for filename in os.listdir(DATASET_DIR1):
        image_path1 = os.path.join(DATASET_DIR1, filename)
        temp = filename.split('_')
        age = int(temp[0])
        image_paths1.append(image_path1)

        if age <= 2:
            target_labels1.append(1)
        elif age <= 11:
            target_labels1.append(2)
        elif age <= 21:
            target_labels1.append(3)
        elif age <= 25:
            target_labels1.append(4)
        elif age <= 27:
            target_labels1.append(5)
        elif age <= 31:
            target_labels1.append(6)
        elif age <= 38:
            target_labels1.append(7)
        elif age <= 48:
            target_labels1.append(8)
        elif age <= 58:
            target_labels1.append(9)
        else:
            target_labels1.append(0)

    dfl = pd.DataFrame()
    dfl['image'], dfl['target'] = image_paths1, target_labels1

    Xl = extract_features(dfl['image'])
    Yl = dfl['target']

    # data shuffel
    # for X and Y
    Z = np.random.permutation(len(X))

    # Z = np.arange(1, Y.shape[0])
    # Z = np.random.shuffle(Z)
    X = X[Z]
    Y = Y[Z]
    # Xs, Ys = sklearn.utils.shuffle(X,Y)
    # for Xl and Yl
    # Zl = np.arange(1, Yl.shape[0])
    # Zl = np.random.shuffle(Zl)
    Zl = np.random.permutation(len(Xl))
    Xl = Xl[Zl]
    Yl = Yl[Zl]
    # if split here
    # return (X[1:13000, :, :, :], Y[1:13000]), (Xl[1:3000, :, :, :], Yl[1:3000])
    return (X, Y), (Xl, Yl)


def extract_features(images):
    features = []
    for image in (images):
        img = load_img(image, color_mode="grayscale")
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape(len(features), 200, 200, 1)
    return features


(x_train, y_train), (x_valid, y_valid) = load_dataset()

print(x_train.shape)
print(y_train.shape)


# flattening


x_train = x_train / 255
x_valid = x_valid / 255

# number of categories
num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)

model = Sequential()
model.add(Conv2D(30, (9, 9), strides=1, padding="same", activation="relu",
                 input_shape=(200, 200, 1)))
model.add(BatchNormalization())
# model.add(MaxPool2D((7, 7), strides=2, padding="same"))
# model.add(Conv2D(15, (9, 9), strides=1, padding="same", activation="relu"))
# model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((7, 7), strides=2, padding="same"))
model.add(Conv2D(8, (9, 9), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((7, 7), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_categories, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", metrics=['accuracy'])

history = model.fit(
    x_train, y_train, epochs=8, verbose=1, validation_data=(x_valid, y_valid)
)


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
