import logging
import json
import os
import glob
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

# encoder-decoder architecture with residual connections and batch normalization
def build_autoencoder_with_residual():
    input_img = Input(shape=(64, 64, 50))
    
    # encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2), padding='same')(x)
    residual_1 = x  # residual connection
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2), padding='same')(x)
    residual_2 = x  # residual connection
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # decoder
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # x = tf.keras.layers.Add()([x, residual_2])
    x += residual_2
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # x = tf.keras.layers.Add()([x, residual_1])
    x += residual_1
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    return autoencoder

logging.basicConfig(filename='/opt/ml/output/data/logs-training.txt', level=logging.DEBUG)


if __name__ == '__main__':
    logging.debug('Hello my custom SageMaker init script!')

    my_model_weights = {
        "yes": [1, 2, 3],
        "no": [4]
    }
    f_output_model = open("/opt/ml/model/my-model-weights.json", "w")
    f_output_model.write(json.dumps(my_model_weights, sort_keys=True, indent=4))
    f_output_model.close()
    logging.debug('model weights dumped to my-model-weights.json')

    f_output_data = open("/opt/ml/output/data/environment-variables.json", "w")
    f_output_data.write(json.dumps(dict(os.environ), sort_keys=True, indent=4))
    f_output_data.close()
    logging.debug('environment variables dumped to environment-variables.json')

    f_output_data = open("/opt/ml/output/data/sys-args.json", "w")
    f_output_data.write(json.dumps(sys.argv[1:], sort_keys=True, indent=4))
    f_output_data.close()
    logging.debug('sys args dumped to sys-args.json')

    f_output_data = open("/opt/ml/output/data/sm-input-dir.json", "w")
    f_output_data.write(json.dumps(glob.glob("{}/*/*/*.*".format(os.environ['SM_INPUT_DIR']))))
    f_output_data.close()
    logging.debug('SM_INPUT_DIR files list dumped to sm-input-dir.json')
    
    
    # load preprocessed data
    images = np.load("{}/data/train/images.npy".format(os.environ['SM_INPUT_DIR']))
    brain_scans = np.load("{}/data/train/brains.npy".format(os.environ['SM_INPUT_DIR']))
    
    autoencoder_with_residual = build_autoencoder_with_residual()
    autoencoder_with_residual.compile(optimizer='sgd', loss='mean_squared_error', metrics=RootMeanSquaredError())

    # callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, mode = 'auto', patience=5, min_lr=0.0001, verbose=1),
        LearningRateScheduler(lambda epoch: 1e-3 * tf.math.exp(-0.1*(epoch//10)))
    ]

    # training
    autoencoder_with_residual.fit(brain_scans, images, 
                                  epochs=50, batch_size=32, 
                                  shuffle=True,
                                  validation_split=0.2)

    model_path = fr"/opt/ml/model/autoencoder.keras"
    # Save the model
    autoencoder_with_residual.save_weights(model_path)


