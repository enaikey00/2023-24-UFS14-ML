import logging
import json
import os
import glob
import sys

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
