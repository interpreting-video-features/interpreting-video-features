## Dependencies
There is a conda environment file, env.yml that can be used to install the necessary environment

## Training
To start training the I3D model, first ensure that the config file in configs/config_i3D_smth.json is correctly updated, as well as the KTH version if required.
Then begin training by running "train_i3d_smth.py -c [config_file] -g [gpus_to_use]" or the corresponding file for KTH.
Any of the following parameters can also be given as arguments and will then be used instead of the defaults from the config:

-lr initial learning rate
-bs batch size
-wd weight decay
-opt optimizer
-sfl shuffle dataset

it is also possible to train the original i3d model by giving the argument
 --msl ""
 which will keep the original strides in the temporal dimension.

## Generating Saliency Maps and Temporal Masks




