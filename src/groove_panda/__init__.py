import os

# Marks this directory as a python package
# Log level for compiled operations of TensorFlow
os.putenv('TF_CPP_MIN_LOG_LEVEL', '2')
# TensorFlow options for more performance but less reproducibility
os.putenv('TF_ENABLE_ONEDNN_OPTS', '1')
os.putenv('TF_USE_CUDNN', '1')
