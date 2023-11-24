import logging
import json
import glob
import sys
from os import environ
from flask import Flask

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

# see https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response
@app.route('/invocations', methods=['POST'])
def invocations():
    logging.debug('Hello from route /invocations')

    model_files = glob.glob("{}/*.*".format('/opt/ml/model'))
    model_files_str = json.dumps(model_files, sort_keys=True, indent=4)
    logging.debug('model_files_str')
    logging.debug(model_files_str)

    for model_file in model_files:
        logging.debug('reading file {}'.format(model_file))
        model_file_content = json.load(open(model_file))
        logging.debug(json.dumps(model_file_content, sort_keys=True, indent=4))

    sys_argv = json.dumps(sys.argv[1:], sort_keys=True, indent=4)
    logging.debug('sys_argv')
    logging.debug(sys_argv)
    
    
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


    autoencoder_with_residual = build_autoencoder_with_residual()
    
    model_path = "/opt/ml/model/autoencoder.keras"
    # Load the model
    autoencoder_with_residual.load_weights(model_path)

    brain_train = brain_scans[1000, :, :, :]
    image_train = images[1000, :, :, :]
    single_brain_scan = np.expand_dims(brain_train, axis=0)
    reconstructed_train = autoencoder_with_residual.predict(single_brain_scan)

    image_train= image_train[:, :, :]
    image_train = ( image_train * 255).astype(np.uint8)

    reconstructed_image = reconstructed_train[0, :, :, :]
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

    print(image_train.shape)
    print(reconstructed_image.shape)

    Image.fromarray(image_train, mode="RGB").save("img.png")
    Image.fromarray(reconstructed_image, mode="RGB").save("img2.png")
    
    bucket_name = "a-random-bucket-name-nik-422723"
    
    import boto3
    s3_resource = boto3.Session().resource('s3')
    s3_resource.Bucket(bucket_name).Object("demo/image_train.png").put(Body="img.png")
    s3_resource.Bucket(bucket_name).Object("demo/image_predicted.png").put(Body="img2.png")



    return {
        'inference_result': 0.5
    }
