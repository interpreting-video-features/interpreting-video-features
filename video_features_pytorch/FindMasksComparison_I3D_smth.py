import os
import sys
import time
import mask
import utils
import torch
import signal
import importlib
import numpy as np
import pandas as pd
import visualisation as viz
import pickle

from grad_cam_videos import GradCamVideo

i3dVer = False
RESIZE_SIZE_WIDTH = 224
RESIZE_SIZE_HEIGHT = 224


# Insert path to the grad cam viz repo https://github.com/jacobgil/pytorch-grad-cam
path_to_grad_cam_repo = "../pytorch-grad-cam/"
if not os.path.exists(path_to_grad_cam_repo):
    raise ValueError("Path to Grad-CAM repo not found. Please correct the path.")

sys.path.insert(0, path_to_grad_cam_repo)

# load configurations
args = utils.load_args()
config = utils.load_module(args.config).config
# set column model
file_name = config['conv_model']
cnn_def = importlib.import_module("{}".format(file_name))

# setup device - CPU or GPU
device, device_ids = utils.setup_cuda_devices(args)

print(" > Using device: {}".format(device.type))

print(" > Active GPU ids: {}".format(device_ids))

best_loss = float('Inf')

if config["input_mode"] == "jpg":
    from data_loader_jpg import ImLoader
else:
    raise ValueError("Please provide a valid input mode")


def main():
    with torch.autograd.detect_anomaly():
        global args, best_loss

        # create model
        model = cnn_def.Model(config['num_classes'],
                              last_stride=1,
                              stride_mod_layers=args.mod_stride_layers,
                              softMax=1)

        # multi GPU setting
        model = torch.nn.DataParallel(model, device_ids).to(device)

        print(" > Using {} processes for data loader.".format(
            config["num_workers"]))

        train_data = ImLoader(config['data_folder'] + "/train/")
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config['batch_size'], shuffle=config["shuffle"],
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=True)

        val_data = ImLoader(config['data_folder'] + "/validation/", get_item_id=True)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=True)

        test_data = ImLoader(config['data_folder'] + "/smth-smth_Data/test/")
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=True)

        # load model from a checkpoint
        checkpoint_path = args.checkpoint

        if os.path.isfile(checkpoint_path):
            # TODO: put back to 'resume' and bring back start epoch maybe
            print(" > Loading checkpoint '{}'".format(checkpoint_path))

            checkpoint = torch.load(checkpoint_path)

            # args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(" > Loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print(" !#! No checkpoint found at '{}'".format(
                checkpoint_path))

        # configure mask gradient descent params
        if args.lam1 is not None:
            lam1 = args.lam1
        else:
            lam1 = 0.01
        if args.lam2 is not None:
            lam2 = args.lam2
        else:
            lam2 = 0.02

        # optIter= how many gradient descent steps to take in finding mask
        if args.optIter is not None:
            N = args.optIter
        else:
            N = 300

        find_masks(val_loader, model, config, lam1, lam2, N, "central", config["maskPerturbType"],
                   classOI=args.subsetFile, doGradCam=True, runTempMask=True)


def find_masks(dat_loader, model, hyper_params, lam1, lam2, N, maskType="gradient", temporalMaskType="freeze",
               classOI=None, verbose=True, maxMaskLength=None, doGradCam=False, runTempMask=True):
    '''
    Finds masks for sequences according to the given maskMode.
    Input:
        dat_loader: iterateble providing input batches (should be val/test)
        model: model to evaulate masks on
        hyper_params: dictionary with lr, weight decay and batchsize that the grad desc method uses to find mask
        lam1: weighting factor for L1 loss
        lam2: weighting factor TV norm loss
        N: amount of iterations to run through when using grad desc method
        ita: number of times to find mask when using grad desc method (useful to eval several rand inits)
        maskMode: How to find the mask. can be one of:
            'combi': iterate through the different combinations of a coherent 'one blob' mask. Does not use grad desc
            'central': initialize an as small as possible centered mask and use grad desc to find optimal mask
            'random': initialize completely random mask and use grad desc to find optimal mask
        classOI: if only specific classes should be evaluated (must be one of the 174 class numbers)
        verbose: Print mask information during grad desc
        temporalMaskType: defines which perturb type is used to find the first mask indexes 
    '''
    model.eval()
    masks = []
    df = pd.read_csv(classOI)
    results_path = 'results/'
    clips_time_mask_results = []
    clips_grad_cam_results = []
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for i, (sequence, label, video_id) in enumerate(dat_loader):
        if i % 50 == 0:
            print("on idx: ", i)

        input_var = sequence.to(device)
        label = label.to(device)

        model.zero_grad()

        # eta is for breaking out of the grad desc early if it hasn't improved
        eta = 0.00001

        for batch_index in range(hyper_params['batch_size']):

            # only look at cases where clip is of a certain class (if class of interest 'classoI' was given)
            true_class = label[batch_index].item()
            true_class_str = str(true_class)

            if (true_class_str in list(df.keys())
                and int(video_id[batch_index]) in [clip for clip in df[true_class_str]])\
                    or classOI is None:

                output = model(input_var)

                if runTempMask:
                    if hyper_params["gradCamType"] == "guessed":
                        mask_target = torch.zeros((hyper_params["batch_size"], 1)).long()
                        mask_target[batch_index] = torch.argmax(output[batch_index])

                    else:
                        mask_target = label
                        # combinatorial mask finding is straight forward, needs no grad desc
                    # gradient descent type
                    model.zero_grad()
                    time_mask = mask.init_mask(
                        input_var, model, batch_index, mask_target, threshold=0.9,
                        mode="central", mask_type=temporalMaskType)
                    optimizer = torch.optim.Adam([time_mask], lr=0.2)
                    oldLoss = 999999
                    for nidx in range(N):

                        if nidx % 25 == 0:
                            print("on nidx: ", nidx)

                        mask_clip = torch.sigmoid(time_mask)
                        l1loss = lam1 * torch.sum(torch.abs(mask_clip))
                        tvnorm_loss = lam2 * mask.calc_tv_norm(mask_clip, p=3, q=3)

                        class_loss = model(mask.perturb_sequence(
                            input_var, mask_clip, perturbation_type=temporalMaskType))

                        class_loss = class_loss[batch_index, mask_target[batch_index]]

                        loss = l1loss + tvnorm_loss + class_loss

                        if abs(oldLoss - loss) < eta:
                            break;

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    time_mask = torch.sigmoid(time_mask)
                    pred_class = int(torch.argmax(output[batch_index]).item())
                    original_score_guess = int(torch.max(output[batch_index]).item())
                    original_score_true = output[batch_index][label[batch_index]].item()
                    video_id = video_id[batch_index]

                    score_save_path = os.path.join(
                        "cam_saved_images", args.subDir, str(true_class),
                        video_id + "g_" + str(pred_class) + "_gs%5.4f" % original_score_guess +
                        "_cs%5.4f" % original_score_true, "combined")

                    if not os.path.exists(score_save_path):
                        os.makedirs(score_save_path)

                    f = open(score_save_path + "/ClassScoreFreezecase" + video_id[batch_index] + ".txt", "w+")
                    f.write(str(class_loss.item()))
                    f.close()

                    class_loss_reverse = model(mask.perturb_sequence(input_var, time_mask, perturbation_type="reverse"))
                    class_loss_reverse = class_loss_reverse[batch_index, mask_target[batch_index]]

                    f = open(score_save_path + "/ClassScoreReversecase" + video_id[batch_index] + ".txt", "w+")
                    f.write(str(class_loss_reverse.item()))
                    f.close()

                    # as soon as you have the time mask, and freeze/reverse scores,
                    # Add results for current clip in list of timemask results
                    clips_time_mask_results.append({'true_class': true_class,
                                                    'pred_class': pred_class,
                                                    'video_id': video_id,
                                                    'time_mask': time_mask.detach().cpu().numpy(),
                                                    'original_score_guess': original_score_guess,
                                                    'original_score_true': original_score_true,
                                                    'freeze_score': class_loss.item(),
                                                    'reverse_score': class_loss_reverse.item()
                                                    })

                if doGradCam:

                    target_index = mask_target[batch_index]

                    grad_cam = GradCamVideo(model=model.module,
                                            target_layer_names=['Mixed_5c'],  # model.module.end_points and ["block5"],
                                            class_dict=None,
                                            use_cuda=True,
                                            input_spatial_size=(RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT),
                                            normalizePerFrame=True,
                                            archType="I3D")
                    input_to_model = input_var[batch_index][None, :, :, :, :]

                    if hyper_params["gradCamType"] == "guessed":
                        target_index = torch.argmax(output[batch_index])

                    mask, output_grad = grad_cam(input_to_model, target_index)

                    # as soon as you have the numpy array of gradcam heatmap, add it to list of GCheatmaps
                    clips_grad_cam_results.append(
                        {'true_class': int(label[batch_index].item()),
                        'pred_class': int(torch.argmax(output[batch_index]).item()),
                        'video_id': int(video_id[batch_index]),
                        'GCHeatMap': mask
                        })

                    '''beginning of gradcam write to disk'''
                    input_data_unnormalised = input_to_model[0].cpu().permute(1, 2, 3, 0).numpy()
                    input_data_unnormalised = np.flip(input_data_unnormalised, 3)

                    targTag = video_id[batch_index]

                    output_images_folder_cam_combined = os.path.join(
                        "cam_saved_images", args.subDir, str(label[batch_index].item()),
                        targTag + "g_" + str(torch.argmax(output[batch_index]).item()) +
                        "_gs%5.4f" % torch.max( output[batch_index]).item() +
                        "_cs%5.4f" % output[batch_index][label[batch_index]].item(),
                         "combined")

                    os.makedirs(output_images_folder_cam_combined, exist_ok=True)

                    RESIZE_FLAG = 0

                if doGradCam and runTempMask:
                    viz.createImageArrays(
                        input_var, mask, time_mask, batch_index, "freeze", output_images_folder_cam_combined,
                        targTag, RESIZE_FLAG, RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT)
                    viz.createImageArrays(
                        input_var, mask, time_mask, batch_index, "reverse", output_images_folder_cam_combined,
                        targTag, RESIZE_FLAG, RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT)

                    masks.append(time_mask)

    # finally, write pickle files to disk
    f = open(results_path + "allTimeMaskResults_" + args.subDir + "_" + classOI + "_" + ".p", "wb")
    pickle.dump(clips_time_mask_results, f)
    f.close()

    f = open(results_path + "allGradCamResults_" + args.subDir + "_" + classOI + "_" + ".p", "wb")
    pickle.dump(clips_grad_cam_results, f)
    f.close()

    return masks


if __name__ == '__main__':
    main()
