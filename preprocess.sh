
IMPROVE_MODEL_SCRIPT=tcnns_preprocess_improve.py

# Set env if CANDLE_MODEL is not in same directory as this script
IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR:-$( dirname -- "$0" )}

python $IMPROVE_MODEL_DIR/$IMPROVE_MODEL_SCRIPT $@
