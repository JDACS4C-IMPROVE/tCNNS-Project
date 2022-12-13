import os
import candle

file_path = os.path.dirname(os.path.realpath(__file__))

# additional definitions
additional_definitions = [
    {
        "name": "drug_conv_width",
        "type": int,
        "nargs": "+",
        "help": "convolution width for each drug convolutional layer",
    },
    {   "name": "drug_conv_out",
        "type": int,
        "nargs": "+", 
        "help": "number of channels for each drug convolutional layer",
    },
    {   "name": "drug_pool",
        "type": int,
        "nargs": "+", 
        "help": "max pooling width and pooling stride for each drug convolutional layer",
    },
    {   "name": "bias_constant",
        "type": float, 
        "help": "value for initializing the bias variable",
    },
    {   "name": "std_dev",
        "type": float, 
        "help": "value for standard deviation parameter of truncated normal distribution related to initializing the weight variable",
    },
    {   "name": "min_loss",
        "type": float, 
        "help": "value for initial minimum loss used for early stopping",
    },
    {   "name": "cell_conv_width",
        "type": int,
        "nargs": "+",
        "help": "convolution width for each cell convolutional layer",
    },
    {   "name": "cell_conv_out",
        "type": int,
        "nargs": "+",
        "help": "number of channels for each cell convolutional layer",
    },
    {   "name": "cell_pool",
        "type": int,
        "nargs": "+", 
        "help": "max pooling width and pooling stride for each cell convolutional layer",
    },
    {   "name": "conv_stride",
        "type": int, 
        "help": "convolution stride",
    }, 
    {   "name": "cell_file",
        "type": str, 
        "help": "file of preprocessed cell data",
    },  
    {   "name": "drug_file",
        "type": str, 
        "help": "file of preprocessed drug data",
    },  
    {   "name": "response_file",
        "type": str, 
        "help": "file of preprocessed response data",
    },
    {   "name": "processed_data",
        "type": str, 
        "help": "file of compressed preprocessed data on FTP site",
    },        
]

# required definitions
required = [
    "data_url",
    "model_name",
    "dense",
    "epochs",
    "batch_size",
    "dropout",
    "learning_rate",
    "output_dir",
    "ckpt_directory"
]

# initialize class
class tCNNS(candle.Benchmark):
    def set_locals(self):
        """
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        """
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definisions = additional_definitions
