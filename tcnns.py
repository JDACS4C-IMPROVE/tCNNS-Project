import os
import candle

file_path = os.path.dirname(os.path.realpath(__file__))

# additional definitions
additional_definitions = [
    {
        "name": "drug_conv1_out",
        "type": int,
        "help": "",
    },
    {   "name": "drug_conv1_pool",
        "type": int, 
        "help": "pooling size for drug convolutional layer 1",
    },
    {   "name": "drug_conv2_out",
        "type": int, 
        "help": "",
    },
    {   "name": "drug_conv2_pool",
        "type": int, 
        "help": "pooling size for drug convolutional layer 2",
    },
    {   "name": "drug_conv3_out",
        "type": int, 
        "help": "",
    },
    {   "name": "drug_conv3_pool",
        "type": int, 
        "help": "pooling size for drug convolutional layer 3",
    },
    {   "name": "cell_conv1_out",
        "type": int, 
        "help": "",
    },
    {   "name": "cell_conv1_pool",
        "type": int, 
        "help": "pooling size for cell convolutional layer 1",
    },
    {   "name": "cell_conv2_out",
        "type": int, 
        "help": "",
    },
    {   "name": "cell_conv2_pool",
        "type": int, 
        "help": "pooling size for cell convolutional layer 2",
    },
    {   "name": "cell_conv3_out",
        "type": int, 
        "help": "",
    },
    {   "name": "cell_conv3_pool",
        "type": int, 
        "help": "pooling size for cell convolutional layer 3",
    },
]

# required definitions
required = [
    "data_url",
    "train_data",
    "test_data",
    "model_name",
    "conv",
    "dense",
    "activation",
    "out_activation",
    "loss",
    "optimizer",
    "feature_subsample",
    "metrics",
    "epochs",
    "batch_size",
    "dropout",
    "classes",
    "pool",
    "output_dir",
]

# initialize class
class tCNNS(candle.Benchmark):
    def set_locals(self):
	"""Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions