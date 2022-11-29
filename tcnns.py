import os
import candle

file_path = os.path.dirname(os.path.realpath(__file__))

# additional definitions
additional_definitions = [
    {
        "name": "drug_conv1_out",
        "type": int,
        "help": "number of channels for drug convolutional layer 1",
    },
    {   "name": "drug_conv2_out",
        "type": int, 
        "help": "number of channels for drug convolutional layer 2",
    },
    {   "name": "drug_conv3_out",
        "type": int, 
        "help": "number of channels for drug convolutional layer 3",
    },
    {   "name": "cell_conv1_out",
        "type": int, 
        "help": "number of channels for cell convolutional layer 1",
    },
    {   "name": "cell_conv2_out",
        "type": int, 
        "help": "number of channels for cell convolutional layer 2",
    },
    {   "name": "cell_conv3_out",
        "type": int, 
        "help": "number of channels for cell convolutional layer 3",
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
    {   "name": "length_smiles",
        "type": int, 
        "help": "number of symbols in longest SMILES",
    },
    {   "name": "num_cell_features",
        "type": int, 
        "help": "number of cell line features",
    },
    {   "name": "conv_width",
        "type": list, 
        "help": "list of integers for convolution width",
    },
    {   "name": "conv_stride",
        "type": int, 
        "help": "convolution stride",
    },
    {   "name": "num_chars_smiles",
        "type": int, 
        "help": "number of unique symbols in SMILES",
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
    "pool"
    "ckpt_directory"
]

# initialize class
class tCNNS(candle.Benchmark):
    def set_locals(self):
	    """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        """
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions