# Hyperparameters

Hyperparameters and other variables of the model can be adjusted in the config file `tcnns_default_model.txt`. 

A different config file can be called by adding the *config* option: 

```sh
export CANDLE_CONFIG=tcnns_benchmark_model.txt
tcnns_train_improve.py --config $CANDLE_CONFIG
```

To get more info on the hyperparameters, refer to [tcnns.py](tcnns.py) or run the following command:
```
tcnns_train_improve.py --help
```

Alternatively, one can modify the hyperparameters on the command line like so:

```sh
tcnns_train_improve.py --epochs 1
```

```diff
+ Do we need a table of hyperparameters
```
