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

    return {
        'inference_result': 0.5
    }
