import os
import sys
import json
import pickle
import argparse
import torch
import shutil
import glob
import importlib


def load_args():
    parser = argparse.ArgumentParser(description='Smth-Smth example training')
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--eval_only', '-e', action='store_true', 
                        help="evaluate trained model on validation data.")
    parser.add_argument('--resume', '-r', action='store_true',
                        help="resume training from a given checkpoint.")
    parser.add_argument('--gpus', '-g', help="GPU ids to use. Please"
                         " enter a comma separated list")
    parser.add_argument('--use_cuda', action='store_true',
                        help="to use GPUs")
    parser.add_argument('--iteration', '-i',
                        help="suffix for model")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help="initial lr")
    parser.add_argument('--batch_size', '-bs', type=int,
                        help="batch size")
    parser.add_argument('--optimizer', '-opt', type=str,
                        help="optimizer to use")
    parser.add_argument('--weight_decay', '-wd', type=float,
                        help="weight decay")
    parser.add_argument('--shuffle', '-sfl', type=int,
                        help="shuffle batch")
    parser.add_argument('--batch_norm', '-bn', type=int,
                        help="use batch_norm or not")
    parser.add_argument('--subDir', '-sd', type=str,
                    help="subdirectory to save figs to")
    parser.add_argument('--dataDir', '-dd', type=str,
                    help="directory containing input data")
    parser.add_argument('--checkpoint', '-chp', type=str,
                    help="checkpoint to read model from")
    parser.add_argument('--train', '-tr', action='store_true',
                    help="to use train data instead of validation")
    parser.add_argument('--lam1', '-l1', type=float,
                    help="L1 loss coeff")
    parser.add_argument('--lam2', '-l2', type=float,
                    help="TV loss coeff")
    parser.add_argument('--maskInitType', '-mi', type=str,
                    help="how to find mask")
    parser.add_argument('--optIter', '-opti', type=int,
                    help="amount of iterations to run mask problem with")
    parser.add_argument('--optRuns', '-optr', type=int,
                    help="amount of different runs to do optimzation for the mask")
    parser.add_argument('--classOI', '-coi', type=int,
                    help="if only a specific class should be considered")
    parser.add_argument('--clstm_hidden', '-chu', type=int,
                    help="number of hidden units clstm")
    parser.add_argument('--clstm_layers', '-chl', type=int,
                    help="number of hidden layers clstm")
    parser.add_argument('--conv_stride', '-ccs', type=int,
                    help="conv stride in clstm")
    parser.add_argument('--final_temp_time', '-ftt', type=int,
                    help="final tempral frame time in i3d")
    parser.add_argument('--last_stride', '-ls', type=int,
                    help="stride on last poolig for I3d")
    parser.add_argument('--mod_stride_layers', '-msl', type=str,
                    help="which I3D layers to modify stride on")
    parser.add_argument('--momentum', '-mom', type=float,
                help="optimizer momentum")
    parser.add_argument('--dropout', '-drop', type=float,
                help="dropout for CLSTM layers")
    parser.add_argument('--num_workers', '-nwork', type=int,
                help="number of workers for dataloader")
    parser.add_argument('--soft_max', '-sm', type=int,
                help="if soft max at end of I3D should be used")
    parser.add_argument('--last_relu', '-lact', type=str,
                help="what activation function at last logit of I3D should be used")
    parser.add_argument('--use_sequence', '-ues', type=int,
                help="if the entire sequence from last CLSTM should be sent to FC")
    parser.add_argument('--gradCamType', '-gct', type=str,
                help="If gradcam and temporal mask should look at guessed or correct classes")
    parser.add_argument('--splitType', '-kths', type=str,
                help="kth split type")
        
    args = parser.parse_args()
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    return args


def remove_module_from_checkpoint_state_dict(state_dict):
    """
    Removes the prefix `module` from weight names that gets added by
    torch.nn.DataParallel()
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def load_json_config(path):
    """ loads a json config file"""
    with open(path) as data_file:
        config = json.load(data_file)
        config = config_init(config)
    return config


def load_module(module_path_and_name):
    # if contained in module it would be a oneliner:
    # config_dict_module = importlib.import_module(dict_module_name)
    module_child_name = module_path_and_name.split('/')[-1].replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_child_name, module_path_and_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config_init(config):
    """ Some of the variables that should exist and contain default values """
    if "augmentation_mappings_json" not in config:
        config["augmentation_mappings_json"] = None
    if "augmentation_types_todo" not in config:
        config["augmentation_types_todo"] = None
    return config


def setup_cuda_devices(args):
    device_ids = []
    device = torch.device("cuda" if args.use_cuda else "cpu")
    if device.type == "cuda":
        device_ids = [int(i) for i in args.gpus.split(',')]
    return device, device_ids


def save_checkpoint(state, is_best, config, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(config['output_dir'], config['model_name'], filename)
    model_path = os.path.join(config['output_dir'], config['model_name'], 'model_best.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        print(" > Best model found at this epoch. Saving ...")
        shutil.copyfile(checkpoint_path, model_path)


def save_results(logits_matrix, features_matrix, targets_list, item_id_list,
                 class_to_idx, config):
    """
    Saves the predicted logits matrix, true labels, sample ids and class
    dictionary for further analysis of results
    """
    print("Saving inference results ...")
    path_to_save = os.path.join(
        config['output_dir'], config['model_name'], "test_results.pkl")
    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, features_matrix, targets_list,
                     item_id_list, class_to_idx], f)


def save_images_for_debug(dir_img, imgs):
    """
    2x3x12x224x224 --> [BS, C, seq_len, H, W]
    """
    print("Saving images to {}".format(dir_img))
    from matplotlib import pylab as plt
    imgs = imgs.permute(0, 2, 3, 4, 1)  # [BS, seq_len, H, W, C]
    imgs = imgs.mul(255).numpy()
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    print(imgs.shape)
    for batch_id, batch in enumerate(imgs):
        batch_dir = os.path.join(dir_img, "batch{}".format(batch_id + 1))
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        for j, img in enumerate(batch):
            plt.imsave(os.path.join(batch_dir, "frame{%04d}.png" % (j + 1)),
                       img.astype("uint8"))


def get_submission(logits_matrix, item_id_list, class_to_idx, config):
    top5_classes_pred_list = []

    for i, id in enumerate(item_id_list):
        logits_sample = logits_matrix[i]
        logits_sample_top5  = logits_sample.argsort()[-5:][::-1]
        # top1_class_index = logits_sample.argmax()
        # top1_class_label = class_to_idx[top1_class_index]

        top5_classes_pred_list.append(logits_sample_top5)

    path_to_save = os.path.join(
            config['output_dir'], config['model_name'], "test_submission.csv")
    with open(path_to_save, 'w') as fw:
        for id, top5_pred in zip(item_id_list, top5_classes_pred_list):
            fw.write("{}".format(id))
            for elem in top5_pred:
                fw.write(";{}".format(elem))
            fw.write("\n")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ExperimentalRunCleaner(object):
    """
    Remove the output dir, if you exit with Ctrl+C and if there are less
    then 1 file. It prevents the noise of experimental runs.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def __call__(self, signal, frame):
        num_files = len(glob.glob(self.save_dir + "/*"))
        if num_files < 1:
            print('Removing: {}'.format(self.save_dir))
            shutil.rmtree(self.save_dir)
        print('You pressed Ctrl+C!')
        sys.exit(0)


# Taken from PyTorch's examples.imagenet.main
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
