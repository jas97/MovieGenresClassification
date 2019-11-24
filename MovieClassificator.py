#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip uninstall keras-preprocessing')
get_ipython().system('pip install git+https://github.com/keras-team/keras-preprocessing.git')
get_ipython().system('pip install keras-metrics')


import numpy as np
import tensorflow as tf
from PIL import Image
from google.colab.patches import cv2_imshow
import os
import cv2
import re
import pandas as pd
import keras
from keras import backend as K
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing.image import load_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras_metrics as km
import matplotlib.pyplot as plt





# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive', force_remount=True)

#parameters
batch_size = 1
save_path = '/content/drive/My Drive/Colab Notebooks/ML projekat/best_model.{epoch:02d}-{val_loss:.2f}.h5'


train_data_dir = '/content/drive/My Drive/Colab Notebooks/ML projekat/data/train_data/train'
validation_data_dir = '/content/drive/My Drive/Colab Notebooks/ML projekat/data/validation_data/validation'
test_data_dir = '/content/drive/My Drive/Colab Notebooks/ML projekat/data/test_data/test'


# show image and prediction for a single image
def predict_genre(model, img):
  # resize for presentation                                  
  plt.imshow(img)
  
  # get probabilities
  probabilities = model.predict(img.reshape(1, 224, 224, 3))
  print(probabilities)
  # keep the best 3
  top_3 = np.argsort(probabilities)[0][-3:]
  
  for i in range(0, 3):
    print(genres[top_3[i]] + '({})'.format(probabilities[0, top_3[i]]))


def get_all_genres(dataset) :
    new_ds = dataset['Genre'].str.split("|", expand=True)

    genres1 = set(new_ds[0])
    genres2 = set(new_ds[1])
    genres3 = set(new_ds[2])

    genres = genres1 | genres2 | genres3

    genres.remove(None)
    genres = list(genres)
    return np.array(genres, dtype=object)


# extracts a year from the movie title
def extractYear(title) :
  try:
    return int(re.search('\(([^)]+)', title).group(1))
  except (ValueError, AttributeError): 
    return None


def genres_to_bool(y):
  y = np.array(y, dtype=object)
  genres_arr = np.array(genres, dtype=object)
  if (len(y) == 1) :
    return np.equal(genres_arr, y[0]) + np.array([0])
  elif (len(y) == 2):
    first = np.equal(genres_arr, y[0])
    second = np.equal(genres_arr, y[1])
    return np.logical_or(first, second) + np.array([0])
  elif (len(y) == 3):
    first = np.equal(genres_arr, y[0])
    second = np.equal(genres_arr, y[1])
    third = np.equal(genres_arr, y[2])
    return np.logical_or(first, np.logical_or(second, third)) + np.array([0])


# returns the number of missclasified labels
def hamming_loss(y, y_hat):
  y_bool = K.cast(y, dtype=bool)
 
  y_hat_bool = y_hat > 0.5
  y_hat_bool = K.cast(y_hat_bool, dtype=bool)
  
  xor = tf.math.logical_xor(y_bool, y_hat_bool)
  
  xor = K.cast(xor, dtype=tf.float32)
 
  return K.mean(xor)


# returns if the top ranked label is not among the proper labels
def one_error_loss(y, y_hat):  
  # sum of mistakes
  mistakes = 0
  # for each instance
  for j in range(0, batch_size): 
    # top ranked label position
    top_ranked_label = tf.math.top_k(y_hat[j], k=1, sorted=True).indices[0]
    # cast to int32
    top_ranked_label = tf.cast(top_ranked_label, dtype=tf.int32)
    # find the proper label on that position
    label = tf.gather(y[j], top_ranked_label)
   
    m = K.switch(K.equal(label, 1), 0, 1)
    mistakes+=m
 
  return mistakes / batch_size  


# returns how far we need to go down the list
# of labels to cover all proper labels
def coverage(y, y_hat) :
  y = K.cast(y, dtype=tf.int32)
    
  # sorted indices of prediction values
  sorted_prediction = tf.math.top_k(y_hat, k=y_hat.shape[-1], sorted=True).indices
    
  sorted_prediction = K.cast(sorted_prediction, dtype=tf.int64)
  
  # sum of distances
  dist_sum = 0;
  
  # for every instance
  for j in range(0, batch_size) :
    
    # position of ones in proper labels
    ones_pos = tf.where(tf.equal(y[j], 1))    

    distances = tf.map_fn(lambda x: tf.where(tf.equal(x, sorted_prediction[j])) + 1, ones_pos)
    dist_sum += K.max(distances)
  
  return dist_sum / batch_size


# read dataset
dataset = pd.read_csv('/content/drive/My Drive/Colab Notebooks/ML projekat/MovieGenre.csv')



# drop all rows with null values
dataset.dropna(inplace=True)


# adds year column
dataset['Year'] = dataset['Title'].apply(extractYear)


# reduce the dataset to movies in the past 10 years
dataset = dataset.drop(dataset[dataset.Year < 2000].index)


# drop duplicates in terms of imdbId
print(dataset.shape)
dataset.drop_duplicates(['imdbId'], keep='last',inplace=True)
print(dataset.shape)


# change imdbId to str
dataset['imdbId'] = dataset['imdbId'].astype(str)
dataset['imdbId'] = dataset['imdbId'] + '.jpg'


# add list of genres as a column
dataset['Genre_modified'] =  dataset['Genre'].str.split("|", expand=False)


# get a numpy array of all possible genres
genres = get_all_genres(dataset)
genres = genres.tolist()
genres.sort()
print(genres)


# create labels column
dataset['Labels'] = dataset['Genre_modified'].apply(genres_to_bool).tolist()
dataset.head(5)


# split labels
dataset = pd.merge(dataset, pd.DataFrame(dataset.Labels.tolist(), columns=genres), on=dataset.index)
dataset.head(5)


# create distribution of genres plot
sums = dataset.iloc[0:][genres].sum(axis=0)
y_pos = np.arange(len(genres))

plt.barh(y_pos, sums, height = 0.8, align='center')
plt.yticks(y_pos, genres)
plt.xlabel('Broj filmova')
plt.title('Distribucija žanrova')

plt.rcParams['figure.figsize'] = (7,7)

plt.show()


# create distribution of number of genres plot
counts = dataset.iloc[0:][genres].sum(axis=1).value_counts()
x_pos = np.array([1, 2, 3])
values = np.array([counts[3], counts[2], counts[1]])

plt.rcParams['figure.figsize'] = (1,1)
plt.bar(x_pos, values, width=0.2, align='center')
plt.xticks(x_pos, x_pos)
plt.ylabel("Broj žanrova")

plt.title("Distribucija broja žanrova")

plt.show()


# data generators
datagen = ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_dataframe(dataframe=dataset, directory=train_data_dir, 
                                            x_col="imdbId", y_col=genres, class_mode='raw',
                                            target_size=(224,224), batch_size=batch_size)

valid_generator=datagen.flow_from_dataframe(dataframe=dataset, directory=validation_data_dir, 
                                            x_col="imdbId", y_col=genres, class_mode="raw",
                                            target_size=(224,224), batch_size=batch_size)

test_generator=datagen.flow_from_dataframe(dataframe=dataset, directory=test_data_dir, 
                                            x_col="imdbId", y_col=genres, class_mode="raw",
                                            target_size=(224,224), batch_size=batch_size)

# model definition
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(genres), activation='sigmoid'))
print(model.summary())


adam = optimizers.Adam(lr=0.0001, clipnorm=1.)
# model compile
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', hamming_loss, one_error_loss, coverage])


mc = ModelCheckpoint(save_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


hist = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=[mc],
                    epochs=10)





# plot losses
plt.rcParams['figure.figsize'] = (7,7)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Treniranje modela')
plt.ylabel('Funkcija gubitka')
plt.xlabel('epoha')
plt.legend(['trening skup', 'validacioni skup'], loc='upper left')
plt.show()





# load the best model
best_model_path = "/content/drive/My Drive/Colab Notebooks/ML projekat/best_model.03-0.21.h5"
saved_model = load_model(best_model_path, custom_objects={ 'hamming_loss': hamming_loss, 'one_error_loss': one_error_loss, 'coverage' : coverage})


# 




# evaluate on the test set
_, test_accuracy, test_hamming_loss, test_one_error_loss, test_coverage = saved_model.evaluate_generator(test_generator, steps = STEP_SIZE_TEST)
print("Testing accuracy: {} \tTesting Hamming loss:{} \tTesting one-error loss: {} \tTesting coverage: {}".format(test_accuracy, test_hamming_loss, test_one_error_loss, test_coverage))







