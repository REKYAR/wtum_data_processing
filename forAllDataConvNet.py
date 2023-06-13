import os
import random
import string
import gc
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
    GlobalAveragePooling2D,
    AvgPool2D,
    SpatialDropout2D
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
        x_batch = []
        y_batch = []
        for b in range(batch_size):
            # end of an epoch - time to supply images from the beginning
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)

            filename = file_list[i]
            image_path = os.path.join(DATASET_DIR, filename)
            temp = filename.split("_")
            age = int(temp[0])
            # y = 0

            # if age <= 2:
            #     y = 0
            # elif age <= 11:
            #     y = 1
            # elif age <= 21:
            #     y = 2
            # elif age <= 25:
            #     y = 3
            # elif age <= 27:
            #     y = 4
            # elif age <= 31:
            #     y = 5
            # elif age <= 38:
            #     y = 6
            # elif age <= 48:
            #     y = 7
            # elif age <= 58:
            #     y = 8
            # else:
            #     y = 9

            i += 1

            y = age
            
            x_batch.append(extract_features(image_path))
            y_batch.append(y)
            
        np_x_batch = np.array(x_batch) / 255   
        # yield tf.convert_to_tensor(np_x_batch), keras.utils.to_categorical(np.array(y_batch), 10)
        yield tf.convert_to_tensor(np_x_batch), np.array(y_batch)
        del np_x_batch
        del x_batch
        del y_batch
        gc.collect()

def extract_features(image_path):
    img = load_img(image_path, color_mode="grayscale")
    img = np.array(img)
    return img


####################### pierwszy plik do uczenia, drugi walidacja ############################

NUM_CATEGORIES = 10

LEARN_DIR = "/media/filip/data/Użytkownicy/filip/Documents/Studia/sem6/Warsztaty z technik uczenia maszynowego/Dataset/dataset_greyscale_full_learn"
VALIDATION_DIR = "/media/filip/data/Użytkownicy/filip/Documents/Studia/sem6/Warsztaty z technik uczenia maszynowego/Dataset/dataset_greyscale_full_validation"

total_learn_files = len(os.listdir(LEARN_DIR))
total_validation_files = len(os.listdir(VALIDATION_DIR))

BATCH_SIZE = 16

print("Constructing model...")
model = Sequential()
model.add(
    Conv2D(
        32,
        (7, 7),
        strides=1,
        activation="relu",
        input_shape=(200, 200, 1),
        padding="same"
    )
)

############################################################
# ctrl+v the model here:

# 83 reg (32,7,7)
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), strides=2, padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.5))

############################################################

model.add(Dense(units=1, activation="linear"))
# model.add(Dense(units=NUM_CATEGORIES, activation="softmax"))
model.summary()

print("Compiling model...")
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.1,
#     decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True
# )
# opt = keras.optimizers.Adam(learning_rate=lr_schedule)

# model = ResNet18(NUM_CATEGORIES)
# model.build(input_shape=(200, 200, 1))
# model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
model.compile(loss="mean_squared_error", metrics=["accuracy"], optimizer="adam")

print("Training model...")
history = model.fit_generator(generator=dataset_generator(LEARN_DIR, BATCH_SIZE), epochs=15,
                    steps_per_epoch=(total_learn_files // BATCH_SIZE),
                    validation_steps=(total_validation_files // BATCH_SIZE),
                    verbose=1, validation_data=dataset_generator(VALIDATION_DIR, BATCH_SIZE))

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

model.save("przygody_w_pociagu.h5", save_format='h5')
print("Saved model to disk")
