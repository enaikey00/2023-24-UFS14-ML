import logging
import json
import glob
import sys
from os import environ
from flask import Flask
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Input, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError, CosineSimilarity, LogCoshError
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import matplotlib.pyplot as plt
import numpy as np
import os
import gzip
import shutil
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import time


if environ.get('AA_LOG_FILE') is not None:
    # only during development we pass this env to log to a file
    logging.basicConfig(filename=environ.get('AA_LOG_FILE'), level=logging.DEBUG)
else:
    # on AWS we should log to the console STDOUT to be able to see logs on AWS CloudWatch
    logging.basicConfig(level=logging.DEBUG)

logging.debug('Init a Flask app')
app = Flask(__name__)


@app.route('/ping')
def ping():
    logging.debug('Hello from route /ping')

    return 'Hello, World!'

print("CIAO")
# ciao
# see https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response

@app.route('/invocations', methods=['POST'])
def invocations():
    print("CIAO2")
    logging.debug('Hello from route /invocations')



    sys_argv = json.dumps(sys.argv[1:], sort_keys=True, indent=4)
    logging.debug('sys_argv')
    logging.debug(sys_argv)
    print("CIO")
    def build_autoencoder_with_residual():
        input_img = Input(shape=(64, 64, 50))
        #input_img = Input(shape=(4272, 64, 64))

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


    autoencoder_with_residual = build_autoencoder_with_residual()
    
    model_path = "/opt/ml/model/autoencoder.keras"
    # Load the model
    autoencoder_with_residual.load_weights(model_path)
    
    image = np.load("/opt/ml/code/image.npy")
    print(image)
    brain_scan = np.load("/opt/ml/code/pippo.npy")
    
    
    np.reshape(brain_scan, (64,64,50))
    #brain_train = brain_scan[1000, :, :, :]
    print(brain_scan.shape)
    #image_train = images[1000, :, :, :]
    single_brain_scan = np.expand_dims(brain_scan, axis=0)
    reconstructed_train = autoencoder_with_residual.predict(single_brain_scan)

    image_train= image[:, :, :]
    image_train = ( image_train* 255).astype(np.uint8)

    reconstructed_image = reconstructed_train[0, :, :, :]
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

    print(image_train.shape)
    print(reconstructed_image.shape)

    Image.fromarray(image_train, mode="RGB").save("img.png")
    Image.fromarray(reconstructed_image, mode="RGB").save("img2.png")
    
    bucket_name = "a-random-bucket-name-nik-301255"
    
    import boto3
    s3_resource = boto3.Session().resource('s3')
    s3_resource.Bucket(bucket_name).Object("demo/image_train.png").put(Body="img.png")
    s3_resource.Bucket(bucket_name).Object("demo/image_predicted.png").put(Body="img2.png")



    return {
        'inference_result': 0.5
    }
    
