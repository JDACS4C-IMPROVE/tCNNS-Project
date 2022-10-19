import candle
import os

# Assumes CANDLE_DATA_DIR is an environment variable
os.environ['CANDLE_DATA_DIR'] = '/tmp/tCNNS/data'

fname='tCNNS_preprocessed_data.tar.gz'
origin='ftp://ftp.mcs.anl.gov/pub/candle/public/improve/tCNNS/tCNNS_preprocessed_data.tar.gz'

# Download and unpack the data in CANDLE_DATA_DIR
candle.file_utils.get_file(fname, origin)

# Do it again to confirm it's not re-downloading
candle.file_utils.get_file(fname, origin)
