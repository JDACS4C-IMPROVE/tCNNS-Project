import os
import candle

file_path = os.path.dirname(os.path.realpath(__file__))

# additional definitions
additional_definitions = [
    {
        "name": "drug_conv_out",
        "type": int,
        "nargs": "+",
        "help": "number of channels for drug convolutional layers",
    },
    {
        "name": "cell_conv_out",
        "type": int,
        "nargs": "+",
        "help": "number of channels for cell convolutional layers",
    },
    {
        "name": "bias_constant",
        "type": float,
        "help": "value for initializing the bias variable",
    },
    {
        "name": "std_dev",
        "type": float,
        "help": "value for standard deviation parameter of truncated normal distribution related to initializing the weight variable",
    },
    {
        "name": "min_loss",
        "type": float,
        "help": "value for initial minimum loss used for early stopping",
    },
    {
        "name": "length_smiles",
        "type": int,
        "help": "number of symbols in longest SMILES",
    },
    {
        "name": "num_cell_features",
        "type": int,
        "help": "number of cell line features",
    },
    {
        "name": "drug_conv_width",
        "type": list,
        "help": "list of integers for convolution width",
    },
    {
        "name": "cell_conv_width",
        "type": list,
        "help": "list of integers for convolution width",
    },
    {
        "name": "drug_pool",
        "type": list,
        "help": "list of integers for drug pool width",
    },
    {
        "name": "cell_pool",
        "type": list,
        "help": "list of integers for cell pool width",
    },
    {
        "name": "conv_stride",
        "type": int,
        "help": "convolution stride",
    },
    {
        "name": "num_chars_smiles",
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
    "drug_pool",
    "cell_pool",
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
