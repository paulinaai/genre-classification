import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

music_train = pd.read_csv('features_30_sec.csv', names = [
        "chroma_stft_mean", "chroma_stft_var", "rms_mean",	"rms_var", "spectral_centroid_mean", 
         "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean", 
         "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var", "harmony_mean", "harmony_var", 
         "perceptr_mean", "perceptr_var", "tempo", "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", 
         "mfcc3_mean", "mfcc3_var", "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", 
         "mfcc6_var", "mfcc7_mean", "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var", 
         "mfcc10_mean", "mfcc10_var", "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var", "mfcc13_mean", 
         "mfcc13_var", "mfcc14_mean", "mfcc14_var", "mfcc15_mean", "mfcc15_var", "mfcc16_mean", "mfcc16_var", 
         "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var", "mfcc19_mean", "mfcc19_var", "mfcc20_mean", 
         "mfcc20_var", "label"
         ])
music_train.head()

music_features = music_train.copy()
music_labels = music_features.pop("label")

music_model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  layers.Dense(57)
])

music_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

music_model.fit(music_features, music_labels, epochs=10)