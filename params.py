import itertools
from torch import nn as nn
import time
import os
from utils import mse, is_class_correct, is_class_different, cross_entropy

params = {
    "batch_size_test"           : 520,
    "numclasses"                : 23,
    "learning_rate"             : 0.001,   # Set by experimentation on 1-2 sample images.
    "cutoff_loss"               : 0.05,    # #Min loss beyond which input optimization is stopped.
    "num_iters"                 : 500,     # #Max iterations for the input optimization process.
    "cuda"                      : True,
    "plotinterim_checkpoint"    : -1,      # Set to -1 to turn off.
    "logdir"                    : 'outputs',# + '_' + str(time.time()),  # Output directory


    "pretrained_model_path"     : os.path.join("model", "spotter_model_23_phrases.pth.tar"),
    "criterion"                 : nn.CrossEntropyLoss(), #loss for input optimiztion. Usually it is the same as the loss function used for model training

    "data_path"                 : os.path.join("data", "test_commands.pt"),
    "transforms"                : {
        "types"                 : ['time_mask', 'freq_mask', 'noise'],
        "values"                : #corresponding parameters for each transform T
            list(itertools.product([0, 30, 60, 120],
                                                         [20, 15, 10, 1, 0],
                                                         [0, 3.0, 15.0, 50.0]
                                          ))},

    "differences_fn"              : 'differences.p', #filename for dump of distances d_x, d_y, d_g, d^_g per input x^T_i

    # compute:
    # 0: difference between inputs (d_x),
    # 1: difference between predicted output classes (d_y),
    # 2: difference between transformed output probability and target (d_g),
    # 3: difference between projected output probability and target (d^_g)
    #-1: change in performance (d_g - d^_g)

    "differences_metrics"          :  { #specify distances to be computed
                                    'mse'   : {'function' : mse                , 'compute': [0, 1]}, #computes mse_x, mse_y
                                    'diff'  : {'function' : is_class_different , 'compute': [1]},
                                    'match' : {'function' : is_class_correct   , 'compute': [2, 3, -1]},
                                    'loss'  : {'function' : cross_entropy      , 'compute': [2, 3, -1]},}
}

def get_params():
    return params