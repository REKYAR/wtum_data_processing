import os
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(gpu_devices)
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# generate data by loading in batches
def dataset_generator(DATASET_DIR : string, batch_size : int):
    """
    Load the CIFAR-10 dataset
    """
    file_list = os.listdir(DATASET_DIR)

    random.shuffle(file_list)

    i = 0

    while True:
        for b in range(batch_size):
            x_batch = []
            y_batch = []
            # end of an epoch - time to supply images from the beginning
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)

            filename = file_list[i]
            image_path = os.path.join(DATASET_DIR, filename)
            temp = filename.split("_")
            age = int(temp[0])
            y = 0

            if age <= 2:
                y = 1
            elif age <= 11:
                y = 2
            elif age <= 21:
                y = 3
            elif age <= 25:
                y = 4
            elif age <= 27:
                y = 5
            elif age <= 31:
                y = 6
            elif age <= 38:
                y = 7
            elif age <= 48:
                y = 8
            elif age <= 58:
                y = 9

            i += 1
            
            x_batch.append(extract_features(image_path))
            y_batch.append(y)
        yield np.array(x_batch) / 255, keras.utils.to_categorical(np.array(y_batch), 10)


def extract_features(image_path):
    img = load_img(image_path, color_mode="grayscale")
    img = np.array(img)
    return img


####################### pierwszy plik do uczenia, drugi walidacja ############################

num_categories = 10

LEARN_DIR = "./dataset_canny_edges_learn"
VALIDATION_DIR = "./dataset_canny_edges_learn"

total_learn_files = len(os.listdir(LEARN_DIR))
total_validation_files = len(os.listdir(VALIDATION_DIR))

batch_size = 32

# model 3 + new conv2d 256
print("Constructing model...")
model = Sequential()
print("Adding the input layer...")
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
print("Adding a MaxPool2D layer...")
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
print("Adding a Conv2D layer...")
model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
print("Adding a MaxPool2D layer...")
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(64, (5, 5), strides=1, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(128, (5, 5), strides=1, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
print("Adding a Dense layer...")
model.add(Dense(units=512, activation="relu"))
print("Adding an output")
model.add(Dense(units=num_categories, activation="softmax"))
model.summary()

print("Compiling model...")
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

print("Training model...")
history = model.fit_generator(generator=dataset_generator(LEARN_DIR, batch_size), epochs=15,
                    steps_per_epoch=(total_learn_files // batch_size),
                    validation_steps=(total_validation_files // batch_size), verbose=1, validation_data=dataset_generator(VALIDATION_DIR, batch_size))

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

model.save("canny_edges")
print("Saved model to disk")
