config = {
    "model_name": "modelI3d_scratch",
    "output_dir": "trained_models/",
    "input_mode": "jpg",

    "data_folder": "/smth_smth/data/smth_smth_orig/20bn-something-something-v2/",

    "json_data_train": "/smth_smth/data/smth_smth_orig/train.json",
    "json_data_val": "/smth_smth/data/smth_smth_orig/validation.json",
    "json_data_test": "/smth_smth/data/smth_smth_orig/test.json",

    "json_file_labels": "/smth_smth/data/smth_smth_orig/labels.json",

    "num_workers": 2,

    "num_classes": 174,
    "batch_size": 16,
    "clip_size": 16,

    "nclips_train": 1,
    "nclips_val": 1,

    "upscale_factor_train": 1.4,
    "upscale_factor_eval": 1.0,

    "step_size_train": 1,
    "step_size_val": 1,

    "optimizer": "ADAM",
    "lr": 0.008,
    "last_lr": 0.00001,
    "momentum": 0.9,
    "weight_decay":  0.00001,
    "num_epochs": 1,
    "print_freq": 4,

    "conv_model": "models.I3D_doubled_kth",
    "input_spatial_size": 160,

    "column_units": 512,
    "save_features": True,
    "shuffle": 1,
    "soft_max": 0,
    "last_relu": None,
    "last_stride": 1,
    'final_temp_time': 2,

    'stride_mod_layers': '',  # e.g. '' or 'MaxPool3d_4a_3x3'
    'dropout': 0.5,
    'pretrained_model_path': 'no_ckpt'
}
