# -*- coding: utf-8 -*-
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPool2D, Dropout, LeakyReLU, Conv2DTranspose
from keras import backend as K
from flownet_s.utils import pad, antipad
import tensorflow as tf

def nn_base(input_tensor=None, trainable=False):
    if K.image_dim_ordering == 'th':
        input_shape = (6, None, None)
    else:
        input_shape = (None, None, 6)
        
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    
    conv_1 = Conv2D(64, (7, 7), strides=(2, 2), name='conv_1')(pad(img_input, 3))
    x = LeakyReLU()(conv_1)
    conv_2 = Conv2D(128, (5, 5), strides=(2, 2), name='conv_2')(pad(x, 2))
    x = LeakyReLU()(conv_2)
    conv_3 = Conv2D(256, (5, 5), strides=(2, 2), name='conv_3')(pad(x, 2))   
    x = LeakyReLU()(conv_3)
    conv3_1 = Conv2D(256, (3, 3), name='conv3_1')(pad(x))
    x = LeakyReLU()(conv3_1)
    
    conv_4 = Conv2D(512, (3, 3), strides=(2, 2), name='conv_4')(pad(x))
    x = LeakyReLU()(conv_4)
    conv4_1 = Conv2D(512, (3, 3), name='conv4_1')(pad(x))
    x = LeakyReLU()(conv4_1)
    conv_5 = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), name='conv_5')(pad(x))
    x = LeakyReLU()(conv_5)
    conv5_1 = Conv2D(512, (3, 3), name='conv5_1')(pad(x))
    x = LeakyReLU()(conv5_1)
    
    conv_6 = Conv2D(1024, (3, 3),strides=(2, 2), name='conv_6')(pad(x))
    x = LeakyReLU()(conv_6)
    conv6_1 = Conv2D(1024, (3, 3), activation='relu', name='conv6_1')(pad(x))
    x = LeakyReLU()(conv6_1)
    
    #Start Refinement Network
    #每进行一次操作都会使得分辨率x2
    predict_flow6 = Conv2D(2, (3, 3), name='predict_flow6')(pad(x))
    deconv5 = antipad(Conv2DTranspose(512, (4, 4), strides=(2, 2), name='deconv5')(x))
    upsample_flow6to5 = antipad(Conv2DTranspose(2, (4, 4), strides=(2, 2), name='upsample_flow6to5')(predict_flow6))
    concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)
    
    predict_flow5 = Conv2D(2, (3, 3), name='predict_flow5')(pad(concat5))
    deconv4 = antipad(Conv2DTranspose(256, (4, 4), strides=(2, 2), name='deconv4')(concat5))
    upsample_flow5to4 = antipad(Conv2DTranspose(2, (4, 4), strides=(2, 2), name='upsample_flow5to4')(predict_flow5))
    concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)
    
    predict_flow4 = Conv2D(2, (3, 3), name='predict_flow4')(pad(concat4))
    deconv3 = antipad(Conv2DTranspose(128, (4, 4), strides=(2, 2), name='deconv3')(concat4))
    upsample_flow4to3 = antipad(Conv2DTranspose(2, (4, 4), strides=(2, 2), name='upsample_flow4to3')(predict_flow4))
    concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)
    
    predict_flow3 = Conv2D(2, (3, 3), name='predict_flow3')(pad(concat3))
    deconv2 = antipad(Conv2DTranspose(64, (4, 4), strides=(2, 2), name='deconv2')(concat3))
    upsample_flow3to2 = antipad(Conv2DTranspose(2, (4, 4), strides=(2, 2), name='upsample_flow3to2')(predict_flow3))
    concat2 = tf.concat([conv_2, deconv2, upsample_flow3to2], axis=3)
    
    predict_flow2 = Conv2D(2, (3, 3), name='predict_flow2')(pad(concat2))
    
    #END: Refinement Network
    
    flow = predict_flow2 * 20.0
    flow = tf.image.resize_bilinear(flow, tf.stack[])
    