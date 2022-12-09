import candle
import os

data_dir = os.environ['CANDLE_DATA_DIR'].rstrip('/')
# Assumes CANDLE_DATA_DIR is an environment variable
#os.environ['CANDLE_DATA_DIR'] = '/tmp/tCNNS/data'

fname='tcnns_data_processed.tar.gz'
origin='https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/tCNNS/tcnns_data_processed.tar.gz'

# Download and unpack the data in CANDLE_DATA_DIR
candle.file_utils.get_file(fname, origin, cache_subdir: str = "data_processed")
