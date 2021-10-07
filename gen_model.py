# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:41:49 2021

@author: Tatsuaki
"""
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,MaxPooling2D, concatenate, Input, BatchNormalization 
from tensorflow.keras import Model
import numpy as np

#U-net  
def make_Model01(img_height, img_width):
    
   #入力データの形状の定義
    inputs = Input(shape=(img_height,img_width,3))
    
    
    c1 = Conv2D(20, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(40, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization() (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = BatchNormalization() (c4)
    p4 = MaxPooling2D((2, 2)) (c4)
    
    c5 = Conv2D(320, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = BatchNormalization() (c5)
    p5 = MaxPooling2D((2, 2)) (c5)
    
    c6 = Conv2D(640, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p5)
    c6 = BatchNormalization() (c6)
    p6 = MaxPooling2D((2, 2)) (c6)
    
    mid = Conv2D(1280, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p6)
    
    u7 = Conv2DTranspose(640, (2, 2), strides=(2, 2), padding='same') (mid)
    c7 = concatenate([u7, c6])
    c7 = Conv2D(640, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
    
    u8 = Conv2DTranspose(320, (2, 2), strides=(2, 2), padding='same') (c7)
    c8 = concatenate([u8, c5])
    c8 = Conv2D(320, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
    
    u9 = Conv2DTranspose(160, (2, 2), strides=(2, 2), padding='same') (c8)
    c9 = concatenate([u9, c4], axis=3)
    u9 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
    
    u10 = Conv2DTranspose(80, (2, 2), strides=(2, 2), padding='same') (u9)
    c10 = concatenate([u10, c3], axis=3)
    u10 = Conv2D(80, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c10)
    
    u11 = Conv2DTranspose(40, (2, 2), strides=(2, 2), padding='same') (u10)
    c11 = concatenate([u11, c2], axis=3)
    u11 = Conv2D(40, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c11)
    
    u12 = Conv2DTranspose(20, (2, 2), strides=(2, 2), padding='same') (u11)
    c12 = concatenate([u12, c1], axis=3)
    u12 = Conv2D(20, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c12)
    
    out = Conv2D(1, (1, 1), activation='sigmoid') (u12)
   # print(np.unique(out.numpy()))
    
    
    return Model(inputs=[inputs], outputs=[out])


