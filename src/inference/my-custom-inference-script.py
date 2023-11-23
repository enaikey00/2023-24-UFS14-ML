import logging
import json
import glob
import sys
from os import environ

if environ.get('AA_LOG_FILE') is not None:
    # only during development we pass this env to log to a file
    logging.basicConfig(filename=environ.get('AA_LOG_FILE'), level=logging.DEBUG)
else:
    # on AWS we should log to the console STDOUT to be able to see logs on AWS CloudWatch
    logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    logging.debug('Hello my custom SageMaker init script!')

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
