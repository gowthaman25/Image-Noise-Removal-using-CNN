import numpy as np
import tensorflow as tf
import tensorflow.keras 
from keras.models import load_model,Sequential
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from skimage.measure import compare_psnr, compare_ssim
from keras.layers import Input,BatchNormalization,Subtract,Conv2D,Lambda,Activation
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler

def cnn_model():
    models = Sequential()
    models.add(Conv2D(32,(3,3),strides=(1,1),padding ='same',input_shape=(None,None,1)))
    models.add(LeakyReLU(alpha = 0.3))
    models.add(Conv2D(64,(3,3),strides=(1,1),padding ='same',input_shape=(None,None,1)))
    models.add(LeakyReLU(alpha = 0.3))
    models.add(Conv2D(128,(3,3),strides=(1,1),padding ='same',input_shape=(None,None,1)))
    models.add(LeakyReLU(alpha = 0.3))
    models.add(Conv2D(64,(3,3),strides=(1,1),padding ='same',input_shape=(None,None,1)))
    models.add(Conv2D(32,(3,3),strides=(1,1),padding ='same',input_shape=(None,None,1)))
    models.add(BatchNormalization(axis =-1,epsilon =1e-3))
    models.add(Conv2D(1,(3,3),padding ='same'))

    return models