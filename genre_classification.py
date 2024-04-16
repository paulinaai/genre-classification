import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import tensorflow.keras.utils as utils
import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow.data as data
import tensorflow.image as image
import argparse


train, validation = utils.image_dataset_from_directory(
    'wildcatdata', 
    label_mode = 'categorical',
    image_size = (278, 278),
    seed = 8080,
    validation_split = .30,
    subset = 'both',
)

train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
validation = validation.cache().prefetch(buffer_size = data.AUTOTUNE)

hflip = train.map(lambda x, y: (image.flip_left_right(x), y))
brightness = train.map(lambda x, y: (image.random_brightness(x, max_delta=0.25), y))
hue = train.map(lambda x, y: (image.random_hue(x, max_delta=0.25), y))
train = train.concatenate(hflip)
train = train.concatenate(brightness)
train = train.concatenate(hue)

# VERSION 1: CLass-based model
class Model:
    def __init__(self, input_size):
        self.model = tf.keras.Sequential()
        # depth, frame size are first 2 args
        # first layer of a Sequential Model should get input_shape as arg
        self.model.add(layers.Conv2D(
            10,
            3,
            strides=1,
            activation=activations.relu,
            input_shape = input_size,
            kernel_regularizer=tf.keras.regularizers.L2(),
            padding = 'same',
        ))

        self.model.add(layers.BatchNormalization())

        # self.model.add(layers.MaxPool2D(
        #         pool_size=2,
        #         strides=1,
        #         padding='same',
        # ))

        self.model.add(layers.Conv2D(
            10,
            3,
            strides=1,
            activation=activations.relu,
            input_shape = input_size,
            kernel_regularizer=tf.keras.regularizers.L2(),
            padding = 'same',
        ))

        self.model.add(layers.BatchNormalization())

        # self.model.add(layers.MaxPool2D(
        #         pool_size=2,
        #         strides=1,
        #         padding='same',
        # ))

        self.model.add(layers.Conv2D(
                12, 
                14, 
                strides=4, 
                activation=activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(),
                input_shape=input_size,
        ))
        # size: 67 x 67 x12
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.MaxPool2D(
                pool_size=3,
                strides=2,
        ))
        # size: 32 x 32 x 12
        self.model.add(layers.Conv2D(
                18,
                3,
                strides=1,
                activation=activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(),
        ))
        # size: 30 x 30 x 18
        self.model.add(layers.MaxPool2D(
                pool_size=2,
                strides=2,
        ))
        # size: 15 x 15 x 18
        self.model.add(layers.Flatten())
        # size: 4050
        self.model.add(layers.Dense(1024, activation=activations.relu))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(256, activation=activations.relu))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(64, activation=activations.relu))
        self.model.add(layers.Dense(16, activation=activations.relu))
        #self.model.add(layers.Dropout(0.7))
        # size of last Dense layer MUST match # of classes
        self.model.add(layers.Dense(5, activation=activations.softmax))
        #self.optimizer = optimizers.Adam(learning_rate=0.0002)
        self.lr_scheduler = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0002,
            decay_steps=2600,
            decay_rate=0.5,
        )
        self.optimizer = optimizers.Adam(learning_rate=self.lr_scheduler)
        self.loss = losses.CategoricalCrossentropy() 
        self.model.compile(
                loss = self.loss,
                optimizer = self.optimizer,
                metrics = ['accuracy'],
        )

model = Model((278, 278, 3))

model.model.summary()
#model.model.load_weights("my_model.keras")
m = model.model.fit(
    train,
    batch_size = 32,
    epochs = 100,
    verbose = 1,
    validation_data = validation,
    validation_batch_size = 32,
)

model.model.save_weights('my_model.keras')

with open('graph.json', 'w') as f:
    json.dump(m.history, f)

ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

def parseFile (filename):
    with open(filename, 'r') as inputFile:
        loss = []
        accuracy = []
        vloss = []
        vaccuracy = []
        data = json.load(inputFile)
        loss = data['loss']
        accuracy = data['accuracy']
        vloss = data['val_loss']
        vaccuracy = data['val_accuracy']
    return loss, accuracy, vloss, vaccuracy

loss, accuracy, vloss, vaccuracy = parseFile ('graph.json')
epochs = len(loss)
assert len(loss) == len(accuracy)
assert len(loss) == len(vloss)
assert len(loss) == len(vaccuracy)

fig, axs = plt.subplots(2)
axs[0].plot(np.arange(0, len(loss)), loss, '-r', label='training')
axs[1].plot(np.arange(0, len(accuracy)), accuracy, '-r', label='training')
axs[0].plot(np.arange(0, len(vloss)), vloss, '-g', label='validation')
axs[1].plot(np.arange(0, len(vaccuracy)), vaccuracy, '-g', label='validation')
axs[0].legend(loc='upper left')
axs[1].legend(loc="upper left")
axs[0].set_title('loss')
axs[1].set_title('accuracy')
plt.show()

#model.save_weights('my_model.keras')

#dimensions are a tuple