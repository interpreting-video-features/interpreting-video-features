i3dVer = False

import os
import sys
import importlib

import torch
import pickle
import numpy as np

import mask
import utils
import visualisation as viz

# Insert path to the grad cam viz repo https://github.com/jacobgil/pytorch-grad-cam
path_to_grad_cam_repo = "pytorch-grad-cam/"
if not os.path.exists(path_to_grad_cam_repo):
    raise ValueError("Path to Grad-CAM repo not found. Please correct the path.")

sys.path.insert(0, path_to_grad_cam_repo)
from grad_cam_videos import GradCamVideo

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
    from data_loader_kth import KTHImLoader
else:
    raise ValueError("Please provide a valid input mode")


def main():
    with torch.autograd.detect_anomaly():
        global args, best_loss

        # create model
        if 'I3D' in config['conv_model']:
            model = cnn_def.Model(config['num_classes'], last_stride=1, stride_mod_layers=args.mod_stride_layers, \
                                  softMax=1, finalTimeLength=4, dropout_keep_prob=args.dropout)
        else:
            model = cnn_def.Model(num_classes=6, nb_lstm_units=config['clstm_hidden'], channels=3,
                                  conv_kernel_size=(5, 5), top_layer=True, avg_pool=False,
                                  batch_normalization=config['batch_norm'], lstm_layers=config['clstm_layers'],
                                  step=32, dropout=config['dropout'], conv_stride=config['conv_stride'],
                                  image_size=(160, 120), effective_step=[7, 15, 23, 31])

        # multi GPU setting
        model = torch.nn.DataParallel(model, device_ids).to(device)

        print(" > Using {} processes for data loader.".format(
            config["num_workers"]))

        train_data = KTHImLoader(config['data_folder'] + "/train", clip_size=config["clip_size"])
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config['batch_size'], shuffle=config["shuffle"],
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=True)

        val_data = KTHImLoader(config['data_folder'] + "/test", clip_size=config["clip_size"], get_item_id=True)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=config['batch_size'], shuffle=config["shuffle"],
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=True)

        test_data = KTHImLoader(config['data_folder'] + "/test", clip_size=config["clip_size"])
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=True)

        # load model from a checkpoint
        checkpoint_path = args.checkpoint

        if os.path.isfile(checkpoint_path):
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
        if args.lam1 != None:
            lam1 = args.lam1
        else:
            lam1 = 0.02
        if args.lam2 != None:
            lam2 = args.lam2
        else:
            lam2 = 0.04

        # optIter= how many gradient descent steps to take in finding mask
        if args.optIter != None:
            N = args.optIter
        else:
            N = 100

        ita = 1

        find_masks(val_loader, model, config, lam1, lam2, N, ita, "central", config["maskPerturbType"], classOI=None,
                   doGradCam=True, runTempMask=True)


def find_masks(dat_loader, model, config, lam1, lam2, N, ita, maskType="gradient", temporalMaskType="freeze",
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
        classOI: if only a specific class should be evaluated (must be one of the 174 class numbers)
        verbose: Print mask information during grad desc
        temporalMaskType: defines which perturb type is used to find the first mask indexes 
    '''
    model.eval()
    masks = []
    resultsPath = "results/"
    clipsTimeMaskResults = []
    clipsGradCamResults = []
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    if config["splitType"] == "original":
        clips_of_interest = [["person17", "boxing", "d1", "_1"],
                             ["person17", "boxing", "d2", "_1"],
                             ["person18", "boxing", "d3", "_1"],
                             ["person18", "boxing", "d4", "_1"],
                             ["person17", "handclapping", "d1", "_1"],
                             ["person17", "handclapping", "d2", "_1"],
                             ["person18", "handclapping", "d3", "_1"],
                             ["person18", "handclapping", "d4", "_1"],
                             ["person17", "handwaving", "d1", "_1"],
                             ["person17", "handwaving", "d2", "_1"],
                             ["person18", "handwaving", "d3", "_1"],
                             ["person18", "handwaving", "d4", "_1"],
                             ["person24", "jogging", "d1", "_1"],
                             ["person24", "jogging", "d2", "_1"],
                             ["person25", "jogging", "d3", "_1"],
                             ["person25", "jogging", "d4", "_1"],
                             ["person24", "running", "d1", "_1"],
                             ["person24", "running", "d2", "_1"],
                             ["person25", "running", "d3", "_1"],
                             ["person25", "running", "d4", "_1"],
                             ["person24", "walking", "d1", "_1"],
                             ["person24", "walking", "d2", "_1"],
                             ["person25", "walking", "d3", "_1"],
                             ["person25", "walking", "d4", "_1"],
                             ]
    else:
        clips_of_interest = [["person07", "boxing", "d1", "_1"],
                             ["person07", "boxing", "d2", "_1"],
                             ["person08", "boxing", "d3", "_1"],
                             ["person08", "boxing", "d4", "_1"],
                             ["person07", "handclapping", "d1", "_1"],
                             ["person07", "handclapping", "d2", "_1"],
                             ["person08", "handclapping", "d3", "_1"],
                             ["person08", "handclapping", "d4", "_1"],
                             ["person07", "handwaving", "d1", "_1"],
                             ["person07", "handwaving", "d2", "_1"],
                             ["person08", "handwaving", "d3", "_1"],
                             ["person08", "handwaving", "d4", "_1"],
                             ["person09", "jogging", "d1", "_1"],
                             ["person09", "jogging", "d2", "_1"],
                             ["person10", "jogging", "d3", "_1"],
                             ["person10", "jogging", "d4", "_1"],
                             ["person09", "running", "d1", "_1"],
                             ["person09", "running", "d2", "_1"],
                             ["person10", "running", "d3", "_1"],
                             ["person10", "running", "d4", "_1"],
                             ["person09", "walking", "d1", "_1"],
                             ["person09", "walking", "d2", "_1"],
                             ["person10", "walking", "d3", "_1"],
                             ["person10", "walking", "d4", "_1"],
                             ]

    for i, (input, target, label) in enumerate(dat_loader):
        if i % 50 == 0:
            print("on idx: ", i)

        input_var = input.to(device)
        target = target.to(device)

        model.zero_grad()

        # eta is for breaking out of the grad desc early if it hasn't improved
        eta = 0.00001

        haveOutput = False
        for intraBidx in range(config["batch_size"]):

            targTag = label[intraBidx]
            tagFound = False

            for coi in clips_of_interest:
                if all([coit in targTag for coit in coi]):
                    tagFound = True

            if tagFound:

                if not haveOutput:
                    output = model(input_var)
                    haveOutput = True

                if runTempMask:
                    if config["gradCamType"] == "guessed":
                        maskTarget = torch.zeros((config["batch_size"], 1)).long()
                        maskTarget[intraBidx] = torch.argmax(output[intraBidx])

                    else:
                        maskTarget = target

                    # gradient descent for finding temporal masks
                    model.zero_grad()
                    timeMask = mask.initMask(input_var, model, intraBidx, maskTarget, thresh=0.9, mode="central",
                                        maskPertType=temporalMaskType)
                    optimizer = torch.optim.Adam([timeMask], lr=0.2)
                    oldLoss = 999999
                    for nidx in range(N):

                        if nidx % 25 == 0:
                            print("on nidx: ", nidx)

                        mask_clip = torch.sigmoid(timeMask)
                        l1loss = lam1 * torch.sum(torch.abs(mask_clip))
                        tvnormLoss = lam2 * mask.calc_TVNorm(mask_clip, p=3, q=3)

                        classLoss = model(mask.perturbSequence(input_var, mask_clip, perbType=temporalMaskType))

                        classLoss = classLoss[intraBidx, maskTarget[intraBidx]]

                        loss = l1loss + tvnormLoss + classLoss

                        if abs(oldLoss - loss) < eta:
                            break;

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    timeMask = torch.sigmoid(timeMask)
                    scoreSavePath = os.path.join("cam_saved_images", args.subDir, str(target[intraBidx].item()), \
                                                 label[intraBidx] + "g_" + str(torch.argmax(output[intraBidx]).item()) \
                                                 + "_gs%5.4f" % torch.max(output[intraBidx]).item() \
                                                 + "_cs%5.4f" % output[intraBidx][target[intraBidx]].item(), "combined")

                    if not os.path.exists(scoreSavePath):
                        os.makedirs(scoreSavePath)

                    f = open(scoreSavePath + "/ClassScoreFreezecase" + label[intraBidx] + ".txt", "w+")
                    f.write(str(classLoss.item()))
                    f.close()

                    classLossFreeze = model(mask.perturbSequence(input_var, timeMask, perbType="reverse"))
                    # classLoss = sm(classLoss)
                    classLossFreeze = classLossFreeze[intraBidx, maskTarget[intraBidx]]

                    f = open(scoreSavePath + "/ClassScoreReversecase" + label[intraBidx] + ".txt", "w+")
                    f.write(str(classLossFreeze.item()))
                    f.close()

                    # as soon as you have the time mask, and freeze/reverse scores,
                    # Add results for current clip in list of timemask results
                    clipsTimeMaskResults.append({'true_class': int(target[intraBidx].item()),
                                                 'pred_class': int(torch.argmax(output[intraBidx]).item()),
                                                 'video_id': label[intraBidx],
                                                 'time_mask': timeMask.detach().cpu().numpy(),
                                                 'original_score_guess': torch.max(output[intraBidx]).item(),
                                                 'original_score_true': output[intraBidx][target[intraBidx]].item(),
                                                 'freeze_score': classLoss.item(),
                                                 'reverse_score': classLossFreeze.item()
                                                 })

                    if verbose:
                        print("resulting mask is: ", timeMask)

                if doGradCam:

                    target_index = maskTarget[intraBidx]

                    RESIZE_SIZE_WIDTH = 160
                    RESIZE_SIZE_HEIGHT = 120

                    grad_cam = GradCamVideo(model=model.module,
                                            target_layer_names=['Mixed_5c'],  # model.module.end_points and ["block5"],
                                            class_dict=None,
                                            use_cuda=True,
                                            input_spatial_size=(RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT),
                                            normalizePerFrame=True,
                                            archType="I3D")
                    input_to_model = input_var[intraBidx][None, :, :, :, :]

                    if config["gradCamType"] == "guessed":
                        target_index = torch.argmax(output[intraBidx])

                    mask, output_grad = grad_cam(input_to_model, target_index)

                    # as soon as you have the numpy array of gradcam heatmap, add it to list of GCheatmaps
                    clipsGradCamResults.append({'true_class': int(target[intraBidx].item()),
                                                'pred_class': int(torch.argmax(output[intraBidx]).item()),
                                                'video_id': label[intraBidx],
                                                'GCHeatMap': mask
                                                })

                    '''beginning of gradcam write to disk'''
                    input_data_unnormalised = input_to_model[0].cpu().permute(1, 2, 3, 0).numpy()
                    input_data_unnormalised = np.flip(input_data_unnormalised, 3)

                    targTag = label[intraBidx]

                    output_images_folder_cam_combined = os.path.join("cam_saved_images", args.subDir,
                                                                     str(target[intraBidx].item()), \
                                                                     targTag + "g_" + str(
                                                                         torch.argmax(output[intraBidx]).item()) \
                                                                     + "_gs%5.4f" % torch.max(output[intraBidx]).item() \
                                                                     + "_cs%5.4f" % output[intraBidx][
                                                                         target[intraBidx]].item(), "combined")

                    os.makedirs(output_images_folder_cam_combined, exist_ok=True)

                    clip_size = mask.shape[0]

                    RESIZE_FLAG = 0
                    SAVE_INDIVIDUALS = 1

                if doGradCam and runTempMask:
                    viz.createImageArrays(input_var, mask, timeMask, intraBidx, "freeze", output_images_folder_cam_combined,
                                      targTag, \
                                      RESIZE_FLAG, RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT)
                    viz.createImageArrays(input_var, mask, timeMask, intraBidx, "reverse",
                                      output_images_folder_cam_combined, targTag, \
                                      RESIZE_FLAG, RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT)

                if runTempMask:
                    viz.vizualize_results(input_var[intraBidx],
                                      mask.perturbSequence(input_var, timeMask, perbType=temporalMaskType)[intraBidx],
                                      timeMask, rootDir=output_images_folder_cam_combined, case=targTag, markImgs=True,
                                      iterTest=False)

                    masks.append(timeMask)

                    # finally, write pickle files to disk

    f = open(resultsPath + "I3d_KTH_allTimeMaskResults_original_" + args.subDir + ".p", "wb")
    pickle.dump(clipsTimeMaskResults, f)
    f.close()

    f = open(resultsPath + "I3d_KTH_allGradCamResults_original_" + args.subDir + ".p", "wb")
    pickle.dump(clipsGradCamResults, f)
    f.close()

    return masks


if __name__ == '__main__':
    main()
