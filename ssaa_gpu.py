import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf


from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image

import glob
import os
from tqdm import tqdm
import warnings;
warnings.filterwarnings('ignore')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# download dummy dataset for testing
cmd1 = 'wget http://vis-www.cs.umass.edu/lfw/lfw.tgz'
os.system(cmd1)

# extract dataset
cmd2 = 'tar -xvzf lfw.tgz'
os.system(cmd2)

#capture paths to images
all_images = glob.glob('lfw/**/*.jpg')


# TODO: add validation for image size
#dummy dataset images are 250p X 250p 
high_quality_images = []
for i in tqdm(all_images):
  #can add target_size = (X, Y, 3) if it take too much computing power
  img = image.load_img(i, target_size = (80, 80, 3))
  img = image.img_to_array(img)
  img = img/255.
  high_quality_images.append(img)


high_quality_images = np.array(high_quality_images)

# split data into train and validation data
train_x, val_x = train_test_split(high_quality_images, random_state=32, test_size=0.1)


# blurr image to simulate low_resolution image
# wont be needed with good dataset, will be replaced by UE base image
# function to reduce image resolution while keeping the image size constant

def pixalate_image(image, scale_percent = 40):
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)

  small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
  # scale back to original size
  width = int(small_image.shape[1] * 100 / scale_percent)
  height = int(small_image.shape[0] * 100 / scale_percent)
  dim = (width, height)

  low_res_image = cv2.resize(small_image, dim, interpolation = cv2.INTER_AREA)

  return low_res_image


# get low resolution images for the training set
train_x_px = []
for i in range(train_x.shape[0]):
  temp = pixalate_image(train_x[i,:,:,:])
  train_x_px.append(temp)

train_x_px = np.array(train_x_px)


# get low resolution images for the validation set
val_x_px = []

for i in range(val_x.shape[0]):
  temp = pixalate_image(val_x[i,:,:,:])
  val_x_px.append(temp)

val_x_px = np.array(val_x_px)


#Build model 
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout

with tf.device('/cpu:0'):
    Input_img = Input(shape=(80, 80, 3))  
    
    #encoding architecture
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(Input_img)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
    x3 = MaxPool2D(padding='same')(x2)
    x4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
    x5 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
    x6 = MaxPool2D(padding='same')(x5)
    encoded = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)
    x6_7 = MaxPool2D(padding='same')(encoded)
    #encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    # # decoding architecture
    x7 = UpSampling2D()(encoded)
    x7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
    x7 = UpSampling2D()(encoded)
    x8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
    x10 = Add()([x5, x9])
    x11 = UpSampling2D()(x10)
    x12 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
    x13 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
    x14 = Add()([x2, x13])
    # x3 = UpSampling2D((2, 2))(x3)
    # x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
    # x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    decoded = Conv2D(3, (3, 3), padding='same',activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x14)
    autoencoder = Model(Input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    autoencoder.summary()

#Train model
with tf.device('/cpu:0'):
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1, mode='auto')

with tf.device('/device:GPU:0'):
    a_e = autoencoder.fit(train_x_px, train_x,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(val_x_px, val_x),
        callbacks=[early_stopper])

from tensorflow.keras.models import Sequential, save_model, load_model
filepath = './saved_model_Y'
save_model(autoencoder, filepath)

autoEncoderModel = load_model(filepath, compile = True)

predictions = autoEncoderModel.predict(val_x_px)
n = 5
plt.figure(figsize= (20,10))
for i in range(n):
  ax = plt.subplot(3, n, i+1)
  plt.imshow(val_x_px[i+5])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)  
  ax = plt.subplot(3, n, i+1+n)
  plt.imshow(predictions[i+5])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

results = autoencoder.evaluate(val_x_px, val_x)
print('val_loss, val_accuracy', results)