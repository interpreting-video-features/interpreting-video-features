config = {
    "model_name": "clstm_v4_",
    "output_dir": "trained_models/",
    "input_mode": "jpg",

    "data_folder": "/Users/sbroome/datasets/KTH/kthDataNewSplit/",
    "num_workers": 2,

    "num_classes": 6,
    "batch_size": 16,
    "clip_size": 32,

    "nclips_train": 1,
    "nclips_val": 1,

    "upscale_factor_train": 1.4,
    "upscale_factor_eval": 1.0,

    "step_size_train": 1,
    "step_size_val": 1,

    "optimizer": "ADAM",
    "lr": 0.008,
    "last_lr": 0.00001,
    "momentum": 0.2,
    "weight_decay": 0.00001,
    "num_epochs": 1,
    "print_freq": 4,

    "conv_model": "models.CLSTM_4",
    "input_spatial_size": 160,

    "column_units": 512,
    "save_features": True,
    "shuffle": 1,
    "soft_max": 0,
    "last_relu": None,
    "last_stride": 1,
    'final_temp_time': 2,
    'clstm_hidden' : 4,
    'clstm_layers' : 2,
    'conv_stride' : 2,
    'batch_norm' : True,
    'dropout' : 0.5,
    'pretrained_model_path': 'no_ckpt',
    'maskPerturbType' : 'freeze',
    'splitType' : 'original',
}
