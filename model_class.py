import os
import numpy as np
from tabnanny import filename_only
import pickle
import math
import argparse
import json
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import tensorflow.keras.utils as utils
from statistics import mean
import matplotlib.pyplot as plt
import tensorflow.data as data
import tensorflow.image as image

# go into open cv and change color into hs

# maybe put in some check points here too

# probably should change around layers (like pooling, but then do i have to fix the math?)
# change the input images to be (233, 233)



train, validation = utils.image_dataset_from_directory(
    'weather', 
    batch_size = 32,
    label_mode = 'categorical',
    image_size = (233, 233), 
    seed = 8888, # by giving the shuffle a seed the shuffle will be the same every
    validation_split = 0.30, # in the paper they did .15
    subset = 'both',
)

# print("Class names:")
# print(train.class_names)

train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
validation = validation.cache().prefetch(buffer_size = data.AUTOTUNE)

# change training from RGB to HSV?? see if it helps. analyze the average hsv values for sandstorm and fogsmog and see if you can amplify their difference
#is helpful when stride is 1
# VERSION 1: Class-based model

hflip = train.map(lambda x, y: (image.flip_left_right(x), y))
train = train.concatenate(hflip)

#BRIGHTNESS AND HUE IF YA WANT BUT I THINK HUE WILL MESS UP THE SANDSTORM

class Model:
    def __init__(self, input_size):
        self.model = tf.keras.Sequential() # based off of tensorflow v10
        # depth, frame size, stride, activation
        # first layer of sequential model should get input_shape as arg
        
        # Input 233 x 233 x 3
        # the math actually doesn't work out
        self.model.add(layers.Conv2D( 
                12, 
                11, 
                strides=3, 
                activation=activations.relu, 
                input_shape=input_size,
                kernel_regularizer=tf.keras.regularizers.L2(),
        )) 
        # Size: 75 x 75 x 12
        
        self.model.add(layers.BatchNormalization()) # whats going on

        self.model.add(layers.MaxPool2D(
                pool_size=3,
                strides=2,   
        ))
        # Size: 37 x 37 x 12
        
        self.model.add(layers.Conv2D(
                20,
                3,
                strides=1, # used to be a stride of 2
                activation=activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(),
        ))
        # Size: 35 x 35 x 16
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.MaxPool2D(
                pool_size=3,
                strides=2,
        ))
        # Size: 17 x 17 x 16

        self.model.add(layers.Conv2D(
                20,
                3,
                strides=1, # used to be a stride of 2
                activation=activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(),
        ))
        # Size: 15 x 15 x 16
        self.model.add(layers.BatchNormalization()) 

        self.model.add(layers.MaxPool2D(
                pool_size=3,
                strides=2, # stride of 1 may be useful or not--lets check
        ))
        # Size: 7 x 7 x 16
        #self.model.add(layers.BatchNormalization())

        self.model.add(layers.Conv2D(
                20,
                3,
                strides=1, # used to be a stride of 2
                activation=activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(),
        ))
        # Size: 
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.MaxPool2D(
                pool_size=3,
                strides=2, # stride of 1 may be useful or not--lets check
        ))
        # Size: 
        #regularizers (like patrick did)
        #conv with size 3 and some padding maybe

        self.model.add(layers.BatchNormalization())


        self.model.add(layers.Flatten())
        # Size: 4624


        self.model.add(layers.Dense(250, activation=activations.relu)) # these numbers kinda sus doe
        #self.model.add(layers.Dropout(0.5)) # alexnet
        self.model.add(layers.Dense(50, activation=activations.relu))
        #self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation=activations.relu))
        # size of last dense layer must match # of classes (do you even know how many classes you have?)
        self.model.add(layers.Dense(5, activation=activations.softmax))
        
        self.lr_scheduler = optimizers.schedules.ExponentialDecay(
            initial_learning_rate= 0.00005, 
            decay_steps=3200, 
            decay_rate=0.1,
        )

        self.optimizer = optimizers.Nadam(learning_rate=0.00001)
        self.loss = losses.CategoricalCrossentropy()
        self.model.compile(
                loss = self.loss, 
                optimizer = self.optimizer,
                metrics = ['accuracy'],
        )



model = Model((233, 233, 3)) # dimensions are a tuple
model.model.summary()

# checkpoint_path = "training_3/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# batch_size = 16

# import math
# n_batches = len(train) / batch_size
# n_batches = math.ceil(n_batches)

# latest = tf.train.latest_checkpoint(checkpoint_dir)
# if latest != None:
#     model.model.load_weights(latest)

# cp_callbacks = [
#     callbacks.ModelCheckpoint(
#         filepath = checkpoint_path,
#         save_weights_only = True,
#         verbose = 2,
#         save_freq=76,
#     )
# ]

# model.model.save_weights(checkpoint_path.format(epoch=0))

m = model.model.fit(
    train, 
    batch_size = 32, 
    epochs = 60, # when 20, reaches .67 but its flattening--possibly can get to .75--might be the best we can do
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

loss, accuracy, vloss, vaccuracy = parseFile('graph.json')
epochs = len(loss)
assert len(loss) == len(accuracy)
assert len(loss) == len(vloss)
assert len(loss) == len(vaccuracy)


fig, axs = plt.subplots(2)
axs[0].plot(np.arange(0, len(loss)), loss, "cornflowerblue", label = 'training')
axs[1].plot(np.arange(0, len(accuracy)), accuracy, "cornflowerblue", label='training')
axs[0].set_title('loss')
axs[0].plot(np.arange(0, len(vloss)), vloss, "deeppink", label ='validation')
axs[1].plot(np.arange(0, len(vaccuracy)), vaccuracy, "deeppink", label='validation')
axs[0].legend(loc='upper left')
axs[1].legend(loc='upper left')
axs[1].set_title('accuracy')
plt.show()


