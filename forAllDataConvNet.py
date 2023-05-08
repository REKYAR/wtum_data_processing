import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def load_dataset(DATASET_DIR, DATASET_DIR1):
    """
    Load the CIFAR-10 dataset
    """

    image_paths = []
    target_labels = []

    for filename in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR, filename)
        temp = filename.split("_")
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
    df["image"], df["target"] = image_paths, target_labels
    X = extract_features(df["image"])
    Y = df["target"]

    # learn data

    image_paths1 = []
    target_labels1 = []

    for filename in os.listdir(DATASET_DIR1):
        image_path1 = os.path.join(DATASET_DIR1, filename)
        temp = filename.split("_")
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
    dfl["image"], dfl["target"] = image_paths1, target_labels1

    Xl = extract_features(dfl["image"])
    Yl = dfl["target"]

    # data shuffel
    Z = np.random.permutation(len(X))

    X = X[Z]
    Y = Y[Z]

    Zl = np.random.permutation(len(Xl))
    Xl = Xl[Zl]
    Yl = Yl[Zl]
    return (X, Y), (Xl, Yl)


def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, color_mode="grayscale")
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape(len(features), 200, 200, 1)
    return features


####################### pierwszy plik do uczenia, drugi walidacja ############################
(x_train, y_train), (x_valid, y_valid) = load_dataset(
    "./generated_dataset_learn_smallset30prim",
    "./generated_dataset_validation_smallset30prim",
)
#############################################################################################

print(x_train.shape)
print(y_train.shape)

# flattening


x_train = x_train / 255
x_valid = x_valid / 255

# number of categories
num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)


# model 3 + new conv2d 256
model = Sequential()
model.add(
    Conv2D(
        32,
        (7, 7),
        strides=1,
        padding="same",
        activation="relu",
        input_shape=(200, 200, 1),
    )
)
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(64, (5, 5), strides=1, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(128, (5, 5), strides=1, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dense(units=num_categories, activation="softmax"))
model.summary()


history = model.fit(
    x_train, y_train, epochs=15, verbose=1, validation_data=(x_valid, y_valid)
)


# Checking the train and test loss and accuracy values from the neural network above.

train_loss = history.history["loss"]
test_loss = history.history["val_loss"]
train_accuracy = history.history["accuracy"]
test_accuracy = history.history["val_accuracy"]

# Plotting a line chart to visualize the loss and accuracy values by epochs.

fig, ax = plt.subplots(ncols=2, figsize=(15, 7))

ax = ax.ravel()

ax[0].plot(train_loss, label="Train Loss", color="royalblue", marker="o", markersize=2)
ax[0].plot(test_loss, label="Test Loss", color="orangered", marker="o", markersize=2)

ax[0].set_xlabel("Epochs", fontsize=14)
ax[0].set_ylabel("Categorical Crossentropy", fontsize=14)

ax[0].legend(fontsize=14)
ax[0].tick_params(axis="both", labelsize=12)

ax[1].plot(
    train_accuracy, label="Train Accuracy", color="royalblue", marker="o", markersize=2
)
ax[1].plot(
    test_accuracy, label="Test Accuracy", color="orangered", marker="o", markersize=2
)

ax[1].set_xlabel("Epochs", fontsize=14)
ax[1].set_ylabel("Accuracy", fontsize=14)

ax[1].legend(fontsize=14)
ax[1].tick_params(axis="both", labelsize=12)


fig.suptitle(
    x=0.5,
    y=0.92,
    t="Lineplots showing loss and accuracy of CNN model by epochs",
    fontsize=16,
)

fig.show()

# Exporting plot image in PNG format.
plt.savefig("./final_cnn_loss_accuracy.png", bbox_inches="tight")


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
