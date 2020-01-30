import tensorflow as tf

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D, Convolution3D
from tensorflow.python.keras.layers import MaxPooling2D, MaxPooling3D
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Cropping2D, Cropping3D

def create_model(img_dims, crop_margin_from_top=80, weight_loss_angle=0.8, weight_loss_throttle=0.2):
    tf.keras.backend.clear_session()

    img_in = Input(shape=(img_dims), name='img_in')

    x = img_in

    x = Cropping2D(((crop_margin_from_top, 0), (0, 0)))(x)

    # Define convolutional neural network to extract features from the images
    x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    # Define decision layers to predict steering and throttle
    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='linear')(x)
    x = Dropout(rate=.5)(x)
    x = Dense(units=50, activation='linear')(x)
    x = Dropout(rate=.5)(x)
    # categorical output of the angle
    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.summary()

    model.compile(optimizer='adam',
                loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                loss_weights={'angle_out': weight_loss_angle,
                                'throttle_out': weight_loss_throttle},
                metrics=['mse', 'mae', 'mape'])

    return model