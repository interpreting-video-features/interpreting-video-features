## Dependencies
There is a conda environment file, env.yml that can be used to install the necessary environment

## Training
To start training the I3D model, first ensure that the config file in configs/config_i3D_smth.json is correctly updated, as well as the KTH version if required.
Then begin training by running 
"train_i3d_smth.py -c [config_file] -g [gpus_to_use] --use_cuda" 
or the corresponding file for KTH.

Any of the following parameters can also be given as arguments and will then be used instead of the defaults from the config:

-lr initial learning rate
-bs batch size
-wd weight decay
-opt optimizer
-sfl shuffle dataset

It is also possible to train the original i3d model by giving the argument
 --msl ""
 which will keep the original strides in the temporal dimension.

## Generating Saliency Maps and Temporal Masks
To generate figures as well as pickle files for the quantitative metrics run (either smth-smth or KTH version)

FindMasksComparison_I3D_smth.py -c [config_file] -g [gpus_to_use] --checkpoint [trained_model_checkpoint] --subDir [root_dir_for_results]

it is also possible to change the following parameters from their defaults

--lam1 (lambda_1 parameter in loss function for temporal masks)
--lam2 (lambda_2 parameter in loss function for temporal masks)
--optIter (iterations of gradient descent to run for temporal masks)

The target class used for GradCam can be changed with --gradCamType, to be either "guessed" or "target", to show saliency maps for the guessed class or correct target class



