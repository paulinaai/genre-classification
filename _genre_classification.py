import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
from tensorflow.keras import Sequential
import tensorflow.keras as keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

VALIDATION_SPLIT = 0.3

df_train = pd.read_csv('features_30_sec.csv', skipinitialspace=True, converters = {'filename': str, 'label': str})

df_train.pop("filename")
df_train.pop("length")
print(df_train.dtypes)
df_train = df_train.iloc[1:]
df_features = df_train.copy()
df_labels = df_features.pop('label')
print(df_train.head()) # just for funsies
for c in df_features:
    df_features[c] = df_features[c].to_numpy()

labels = []
print(df_labels.dtypes)
for x in df_labels:
    if x == "blues":
       labels.append(1.0) 
    elif x == 'classical':
        labels.append(2.0)
    elif x == 'country':
        labels.append(3.0)
    elif x == 'disco':
        labels.append(4.0)
    elif x == 'hiphop':
        labels.append(5.0)
    elif x == 'jazz':
        labels.append(6.0)
    elif x == 'metal':
        labels.append(7.0)
    elif x == 'pop':
        labels.append(8.0)
    elif x == 'reggae':
        labels.append(9.0)
    elif x == 'rock':
        labels.append(10.0)

labels = pd.Series(labels)
print(labels) # these are the labels for the music

X_train, X_val, Y_train, Y_val = train_test_split(df_features, labels, test_size=VALIDATION_SPLIT)
#X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

X_train = np.asarray(X_train).astype(np.float32)
Y_train = np.asarray(Y_train).astype(np.float32)

#inputs = keras.Input(shape=(31,))

print("x_train:")
print(X_train)
X_train = tf.convert_to_tensor(X_train) 
Y_val = tf.one_hot(Y_val, 10)

X_val = tf.convert_to_tensor(X_val)

print("y_train:")
Y_train = tf.one_hot(Y_train, 10)
print(Y_train)


df_model = tf.keras.Sequential()
#df_model.add()
df_model.add(layers.Dense(60, activation=activations.relu))
df_model.add(layers.Dense(20, activation=activations.relu))
df_model.add(layers.Dense(15, activation=activations.relu))
df_model.add(layers.Dense(10, activation=activations.sigmoid))
optimizer = optimizers.legacy.Adam(learning_rate = 0.00001) #0.00001
loss = losses.CategoricalCrossentropy()

df_model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = ['accuracy'],
)

# res = df_model.fit(dataset, epochs = 5)
df_model.fit(
    X_train,
    Y_train,
    batch_size = 32,
    epochs = 50,
    validation_data = (X_val, Y_val),
)

# df_features = np.array(df_features)

# for c in df_features:
#   df_features[c] = df_features[c].astype(np.float32) # probably where the error occured
# print(df_features)

# make a function that makes a row from a dataframe into  a tuple with the first item being the entire row except for the last item and the second item being the last item in the row's data
# def dfToTuple(row): # row = row from dataframe (since we r mapping somehow..) (is the dtype of row a df?)
#     df1 = pd.DataFrame[row:"label"]
#     df2 = 2
#     newTuple = (df1, df2)
#     return newTuple

# tupleList = []
# for index, row in df.iterrows():
#     print(row)
#     if index == 10: 
#         quit()
#     # newTuple = 
#     # tupleList += newTuple

    

# # music_features = music_train.copy()

# train, validation = utils.text_dataset_from_directory

# music_model = tf.keras.Sequential([
#   layers.Dense(64, activation='relu'),
#   layers.Dense(1)
# ])

# music_model.compile(loss = tf.keras.losses.MeanSquaredError(),
#                       optimizer = tf.keras.optimizers.Adam())

# music_model.fit(
#     train,
#     validation_data = valid,
#     epochs = 10,
# )


# dataset = tf.data.experimental.make_csv_dataset(
#     "features_30_sec.csv",
#     batch_size=10,
#     field_delim=",",
#     num_epochs=1,
#     select_columns=["chroma_stft_mean", "chroma_stft_var", "rms_mean",	"rms_var", "spectral_centroid_mean", 
#          "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean", 
#          "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var", "harmony_mean", "harmony_var", 
#          "perceptr_mean", "perceptr_var", "tempo", "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", 
#          "mfcc3_mean", "mfcc3_var", "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", 
#          "mfcc6_var", "mfcc7_mean", "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var", 
#          "mfcc10_mean", "mfcc10_var", "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var", "mfcc13_mean", 
#          "mfcc13_var", "mfcc14_mean", "mfcc14_var", "mfcc15_mean", "mfcc15_var", "mfcc16_mean", "mfcc16_var", 
#          "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var", "mfcc19_mean", "mfcc19_var", "mfcc20_mean", 
#          "mfcc20_var", "label"],
#     label_name='label')

# for x in dataset:
#     print(x)


# names = ["filename", "length",
#         "chroma_stft_mean", "chroma_stft_var", "rms_mean",	"rms_var", "spectral_centroid_mean", 
#          "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean", 
#          "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var", "harmony_mean", "harmony_var", 
#          "perceptr_mean", "perceptr_var", "tempo", "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", 
#          "mfcc3_mean", "mfcc3_var", "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", 
#          "mfcc6_var", "mfcc7_mean", "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var", 
#          "mfcc10_mean", "mfcc10_var", "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var", "mfcc13_mean", 
#          "mfcc13_var", "mfcc14_mean", "mfcc14_var", "mfcc15_mean", "mfcc15_var", "mfcc16_mean", "mfcc16_var", 
#          "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var", "mfcc19_mean", "mfcc19_var", "mfcc20_mean", 
#          "mfcc20_var", "label"
#          ]