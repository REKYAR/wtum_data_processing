# 150 reg (32, 3,3)
# model.add(MaxPool2D((2, 2), strides=2))
# model.add(Conv2D(32, (3, 3), strides=1, activation="relu"))
# model.add(Conv2D(64, (3, 3), strides=1, activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2))
# # model.add(Dropout(0.1))
# # model.add(Conv2D(64, (3, 3), strides=1, activation="relu"))
# # model.add(MaxPool2D((2, 2), strides=2))
# model.add(Flatten())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(units=16, activation="relu"))

# # 180 reg (32, 3,3)
# model.add(MaxPool2D(2, 2)),
# # The second convolution
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPool2D(2,2))
# model.add(Flatten())
# # 512 neuron hidden layer
# model.add(Dense(512, activation='relu'))

# # 100 reg (32, 7,7)
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.3))
# model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
# model.add(Conv2D(64, (3, 3), strides=2, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(units=256, activation="relu"))
# model.add(Dropout(0.4))

# # 105 reg (32,7,7)
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
# # model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (5, 5), strides=1, padding="same", activation="relu"))
# model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(Conv2D(64, (3, 3), strides=2, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.5))

# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
# # model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(Conv2D(32, (3, 3), strides=2, padding="same", activation="relu"))
# model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.5))

# # 83 reg (32,7,7)
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(Conv2D(128, (3, 3), strides=2, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.5))

# # https://arxiv.org/pdf/2110.12633.pdf
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=1, padding="same"))
# model.add(Dense(units=256, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(units=128, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(units=64, activation="relu"))
# model.add(Dropout(0.5))


# dobre wyniki dla train, złe dla val
# model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(Conv2D(128, (3, 3), strides=2, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.5))

# 98 val + dobry na train (25 e)
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(128, (3, 3), strides=2, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.1))
# model.add(Flatten())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.5))

# reg 97 - wykres 97
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.6))
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.6))

# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.6))
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.6))


# Abhishek Sharma (class (32, 3, 3))
# model.add(AvgPool2D((2, 2), padding="same"))
# model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(AvgPool2D((2, 2), padding="same"))
# model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(AvgPool2D((2, 2), padding="same"))
# model.add(Dropout(0.1))
# model.add(Flatten())
# model.add(Dense(units=128, activation="relu"))
# model.add(Dropout(0.3))


# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPool2D(2,2))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))

# model.add(Conv2D(32, (3, 3), activation='relu')),
# model.add(Conv2D(32, (3, 3), activation='relu')),
# model.add(MaxPool2D((2,2))),
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3, 3), activation='relu')),
# model.add(Conv2D(64, (3, 3), activation='relu')),
# model.add(MaxPool2D((2,2))),
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (3, 3), activation='relu')),
# model.add(Conv2D(128, (3, 3), activation='relu')),
# model.add(Flatten()),
# model.add(Dense(units=512, activation='relu')),
# model.add(Dropout(0.3))
# model.add(Dense(units=256, activation='relu')),
# model.add(Dropout(0.3))
# model.add(Dense(units=64, activation='relu')),
# model.add(Dropout(0.1))

# model.add(Conv2D(16, (3, 3), activation='relu')),
# # model.add(Conv2D(32, (3, 3), activation='relu')),
# model.add(MaxPool2D((2,2))),
# # model.add(Dropout(0.2))
# model.add(Conv2D(32, (3, 3), activation='relu')),
# # model.add(Conv2D(64, (3, 3), activation='relu')),
# model.add(MaxPool2D((2,2))),
# model.add(Dropout(0.1))
# # model.add(Conv2D(32, (3, 3), activation='relu')),
# # model.add(Conv2D(128, (3, 3), activation='relu')),
# model.add(GlobalAveragePooling2D())
# model.add(Dense(units=64, activation='relu')),
# model.add(Dropout(0.1))
# model.add(Dense(units=64, activation='relu')),
# model.add(Dropout(0.1))
# # model.add(Dense(units=64, activation='relu')),
# # model.add(Dropout(0.1))