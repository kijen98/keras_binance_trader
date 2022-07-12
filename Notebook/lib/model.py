import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras import backend as K

init = 'he_normal'
reg = regularizers.l1(1e-4)
reg = regularizers.l2(1e-4)
reg = None
#l1_l2reg = regularizers.l1_l2(l1=1e-2, l2=1e-2)

def sign_acc(y_true, y_pred):
    return tf.math.floor(K.clip(y_true * y_pred, -0.5, 0.5)) + 1

def inception(data_shapes, model_configs):
    X_shape, y_shape = data_shapes
    optimizer, loss, metrics = model_configs

    input = keras.Input(shape=X_shape)
    mul = 32

    i1 = layers.Conv1D(mul*2, kernel_size=1, strides=2, activation=None, padding='same', kernel_initializer=init)(input)
    i1 = layers.BatchNormalization()(i1)
    i1 = layers.ReLU()(i1)

    i2 = layers.Conv1D(mul*1, kernel_size=1, strides=1, activation=None, padding='same', kernel_initializer=init)(input)
    i2 = layers.BatchNormalization()(i2)
    i2 = layers.ReLU()(i2)
    i2 = layers.Conv1D(mul*2, kernel_size=3, strides=2, activation=None, padding='same', kernel_initializer=init)(i2)
    i2 = layers.BatchNormalization()(i2)
    i2 = layers.ReLU()(i2)

    i3 = layers.AveragePooling1D(pool_size=3, strides=2, padding='same')(input)
    i3 = layers.Conv1D(mul*2, kernel_size=3, strides=1, activation=None, padding='same', kernel_initializer=init)(i3)
    i3 = layers.BatchNormalization()(i3)
    i3 = layers.ReLU()(i3)

    i4 = layers.Conv1D(mul*1, kernel_size=1, strides=1, activation=None, padding='same', kernel_initializer=init)(input)
    i4 = layers.BatchNormalization()(i4)
    i4 = layers.ReLU()(i4)
    i4 = layers.Conv1D(mul*1, kernel_size=3, strides=1, activation=None, padding='same', kernel_initializer=init)(i4)
    i4 = layers.BatchNormalization()(i4)
    i4 = layers.ReLU()(i4)
    i4 = layers.Conv1D(mul*2, kernel_size=3, strides=2, activation=None, padding='same', kernel_initializer=init)(i4)
    i4 = layers.BatchNormalization()(i4)
    i4 = layers.ReLU()(i4)

    x = layers.Concatenate()([i1, i2, i3, i4])
    x = layers.Flatten()(x)

    x = layers.Dense(mul*4, activation=None, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(mul*2, activation=None, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(mul*1, activation=None, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(y_shape[-1], activation='linear', kernel_initializer=init)(x)

    predictor = keras.models.Model(input, x)
    predictor.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return predictor

def lstm(data_shapes, model_configs):
    X_shape, y_shape = data_shapes
    optimizer, loss, metrics = model_configs
    
    mul = 1024
    drop_rate = 0.2
    
    input = keras.Input(shape=X_shape)
    x = layers.LSTM(mul, dropout=drop_rate, return_sequences=False)(input)
    x = layers.Dense(y_shape[-1], activation='linear')(x)

    predictor = keras.models.Model(input, x)
    predictor.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return predictor

def vgg(data_shapes, model_configs):
    X_shape, y_shape = data_shapes
    optimizer, loss, metrics = model_configs
    
    vgg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
           512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    dense = [4096, 4096, 1024]

    div = 1
    input = keras.Input(shape=X_shape)

    x = input
    for dim in vgg:
        if dim == 'M':
            #x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
            x = layers.Conv1D(int(prev_dim/div), kernel_size=5, strides=2, activation=None, padding='same', kernel_initializer=init, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('elu')(x)
            x = layers.Dropout(0.3)(x)
        else:
            x = layers.Conv1D(int(dim/div), kernel_size=5, strides=1, activation=None, padding='same', kernel_initializer=init, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('elu')(x)
            x = layers.Dropout(0.3)(x)
        prev_dim = dim
    #x = layers.Conv1D(1, kernel_size=4, strides=1, activation='linear', kernel_initializer=init, kernel_regularizer=reg)(x)
    x = layers.Flatten()(x)
    
    for dim in dense:
        x = layers.Dense(int(dim/div), activation=None, kernel_initializer=init, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(y_shape[-1], activation='linear', kernel_initializer=init, kernel_regularizer=reg)(x)
    
    predictor = keras.models.Model(input, x)
    predictor.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return predictor

def dense(data_shapes, model_configs):
    X_shape, y_shape = data_shapes
    optimizer, loss, metrics = model_configs
    
    dense = [1024, 2048, 4096, 2048, 1024, 512, 256, 128]
    div = 1

    input = keras.Input(shape=X_shape)
    x = layers.Flatten()(input)
    
    for dim in dense:
        x = layers.Dense(int(dim/div), activation=None, kernel_initializer=init, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ELU()(x)
        x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(y_shape[-1], activation='linear', kernel_initializer=init, kernel_regularizer=reg)(x)
    
    predictor = keras.models.Model(input, x)
    predictor.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return predictor

def autoencoder(data_shapes, model_configs):
    X_shape, y_shape = data_shapes
    optimizer, loss, metrics = model_configs
    
    conv = [32, 64, 128, 256, 512]
    input = keras.Input(shape=X_shape)
    x = input
    for dim in conv:
        x = layers.Conv1D(dim, kernel_size=3, strides=2, activation=None, padding='same', kernel_initializer=init, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        #x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(1024, activation='tanh', kernel_initializer=init, kernel_regularizer=reg)(x)
    rate = layers.Dense(1, activation='linear', kernel_initializer=init, kernel_regularizer=reg)(x)
    
    deconv = [256, 128, 64, 32]
    x = layers.Reshape((1, 1024))(latent)
    x = layers.Conv1DTranspose(512, kernel_size=4, strides=1, activation=None, kernel_initializer=init, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    for dim in deconv:
        x = layers.Conv1DTranspose(dim, kernel_size=3, strides=2, activation=None, padding='same', kernel_initializer=init, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
    x = layers.Conv1DTranspose(5, kernel_size=3, strides=2, activation='linear', padding='same', kernel_initializer=init, kernel_regularizer=reg)(x)
    
    predictor = keras.models.Model(input, [x, rate])
    predictor.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return predictor