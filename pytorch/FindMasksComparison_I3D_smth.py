i3dVer=False

from comet_ml import Experiment
import os
import sys
import time
import signal
import importlib

import torch
import torch.nn as nn
from torch import autograd
import numpy as np

from utils import *
from callbacks import (PlotLearning, AverageMeter)
from models.multi_column import MultiColumn
import torchvision
from transforms_video import *
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from PIL import Image

# Insert path to the grad cam viz repo https://github.com/jacobgil/pytorch-grad-cam
path_to_grad_cam_repo = "../pytorch-grad-cam/"
if not os.path.exists(path_to_grad_cam_repo):
    raise ValueError("Path to Grad-CAM repo not found. Please correct the path.")
    
sys.path.insert(0, path_to_grad_cam_repo)
from grad_cam_videos import GradCamVideo

# load configurations
args = load_args()
config=load_json_config(args.config)
# set column model
file_name = config['conv_model']
cnn_def = importlib.import_module("{}".format(file_name))

# setup device - CPU or GPU
device, device_ids = setup_cuda_devices(args)

print(" > Using device: {}".format(device.type))


print(" > Active GPU ids: {}".format(device_ids))

best_loss = float('Inf')

if config["input_mode"] == "av":
    from data_loader_av import VideoFolder
elif config["input_mode"] == "skvideo":
    from data_loader_skvideo import VideoFolder
elif config["input_mode"] == "jpg":
    from data_loader_jpg import ImLoader
else:
    raise ValueError("Please provide a valid input mode")

hyper_params = {
    "clip_size": config["clip_size"],
    "batch_size": config["batch_size"],
    "num_workers": config["num_workers"],
    "optimizer": config["optimizer"],
    "weight_decay": config["weight_decay"],
    "lr": config["lr"],
    "last_lr": config["last_lr"],
    "momentum": config["momentum"],
    "input_spatial_size": config["input_spatial_size"],
    "column_units": config["column_units"],
    "num_classes": config["num_classes"],
    "shuffle":1
}

#some things are just easier to give via cmdline than changing conf all the time
if(not args.maskInitType==None):
    hyper_params["maskPerturbType"] = args.maskInitType
else:
    hyper_params["maskPerturbType"] = "freeze"
if(not args.batch_size==None):
    hyper_params["batch_size"] = args.batch_size
if(not args.learning_rate==None):
    hyper_params["lr"] = args.learning_rate
if(not args.weight_decay==None):
    hyper_params["weight_decay"] = args.weight_decay
if(not args.optimizer==None):
    hyper_params["optimizer"] = args.optimizer
if(not args.shuffle==None):
    hyper_params["shuffle"] = args.shuffle
if(not args.gradCamType==None):
    hyper_params["gradCamType"] = args.gradCamType
else:
    hyper_params["gradCamType"]="guessed"
    
print("batch size", hyper_params["batch_size"])

def main():
    with autograd.detect_anomaly():
        global args, best_loss

        # create model
        model = cnn_def.Model(config['num_classes'],last_stride=1,stride_mod_layers = args.mod_stride_layers, softMax=1)


        # multi GPU setting
        model = torch.nn.DataParallel(model, device_ids).to(device)

        print(" > Using {} processes for data loader.".format(
            hyper_params["num_workers"]))


        train_data = ImLoader("/smth-smth_Data/train/")
        train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=hyper_params['batch_size'], shuffle=hyper_params["shuffle"],
                num_workers=hyper_params['num_workers'], pin_memory=True,
                drop_last=True)

        val_data = ImLoader("/smth-smth_Data/validation/", get_item_id=True)
        val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=hyper_params['batch_size'], shuffle=False,
                num_workers=hyper_params['num_workers'], pin_memory=True,
                drop_last=True)

        test_data = ImLoader("/smth-smth_Data/test/")
        test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=hyper_params['batch_size'], shuffle=False,
                num_workers=hyper_params['num_workers'], pin_memory=True,
                drop_last=True)


        #load model from a checkpoint
        checkpoint_path=args.checkpoint

        if os.path.isfile(checkpoint_path):
            #TODO: put back to 'resume' and bring back start epoch maybe
            print(" > Loading checkpoint '{}'".format(checkpoint_path))

            checkpoint = torch.load(checkpoint_path)

            #args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(" > Loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print(" !#! No checkpoint found at '{}'".format(
                checkpoint_path))

        
        #configure mask gradient descent params
        if(args.lam1!=None):
            lam1=args.lam1
        else:
            lam1 = 0.01
        if(args.lam2!=None):
            lam2=args.lam2
        else:
            lam2 = 0.02
        
        #optIter= how many gradient descent steps to take in finding mask
        if(args.optIter!=None):
            N=args.optIter
        else:
            N = 300
        
         ita = 1

        find_masks(val_loader,model,hyper_params,lam1, lam2, N, ita, "central", hyper_params["maskPerturbType"], classOI=args.subsetFile, doGradCam=True, runTempMask=True)

def perturbSequence(seq, mask, perbType='freeze', snapValues=False):

    if(snapValues):
        for j in range(len(mask)):
            if(mask[j]>0.5):
                mask[j]=1
            else:
                mask[j]=0
    if (perbType=='freeze'):
        #pytorch expects Batch,Channel, T, H, W
        perbInput = torch.zeros_like(seq)# seq.clone().detach()
        #TODO: also double check that the mask here is not a copy but by ref so the grad holds 
        for u in range(len(mask)):
            
            if(u==0):
                perbInput[:,:,u,:,:]=seq[:,:,u,:,:]
            if(u!=0):
                perbInput[:,:,u,:,:]=(1-mask[u])*seq[:,:,u,:,:] + mask[u]*perbInput.clone()[:,:,u-1,:,:]

    if (perbType=='reverse'):
        #pytorch expects Batch,Channel, T, H, W
        perbInput = torch.zeros_like(seq)
        
        #threshold for finding out which indexes are on
        maskOnInds = (mask>0.1).nonzero()
        if(len(maskOnInds)>0):
            
            #non-zero gives unfortunate dimensions like [[0], [1]] so squeeze it
            maskOnInds = maskOnInds.squeeze(dim=1)
        maskOnInds = maskOnInds.tolist()
        
        subMasks = findSubMasksFromMask(mask,0.1)
        
        for y in range(len(mask)):
            #start with original
            perbInput[:,:,y,:,:]=seq[:,:,y,:,:]
                
        for maskOnInds in subMasks:
            '''if the submask is on at 3,4,5,6,7. the point is to go to the almost halfway point
            as in 3,4 (so that's why it's len()//2) and swap it with the u:th last element instead (like 7,6)
            '''
            for u in range(int(len(maskOnInds)//2)):
                #temp storage when doing swap
                temp = seq[:,:,maskOnInds[u],:,:].clone()
                #first move u:th LAST frame to u:th frame
                perbInput[:,:,maskOnInds[u],:,:]=(1-mask[maskOnInds[u]])*seq[:,:,maskOnInds[u],:,:] + mask[maskOnInds[u]]*seq[:,:,maskOnInds[-(u+1)],:,:]
                
                #then use temp storage to move u:th frame to u:th last frame
                perbInput[:,:,maskOnInds[-(u+1)],:,:]=(1-mask[maskOnInds[u]])*seq[:,:,maskOnInds[-(u+1)],:,:] + mask[maskOnInds[u]]*temp    
    #print("return type of pertb: ", perbInput.type())
    return perbInput

def findSubMasksFromMask(mask, thresh=0.1):
    subMasks = []
    currentSubMask = []
    currentlyInMask = False
    
    for j in range(len(mask)):
        
        #if we find a value above threshold but is first occurence, start appending to current submask
        if(mask[j]>thresh and not currentlyInMask):
            currentSubMask = []
            currentlyInMask=True
            currentSubMask.append(j)
        #if it's not current occurence, just keep appending
        elif(mask[j]>thresh and currentlyInMask):
            currentSubMask.append(j)
            
        #if below thresh, stop appending
        elif((mask[j]<=thresh and currentlyInMask)):
            subMasks.append(currentSubMask)
            currentlyInMask=False
            
        #if at the end of clip, create last submask
        if(j==len(mask)-1 and currentlyInMask):
            subMasks.append(currentSubMask)
            currentlyInMask=False
    #print("submasks found: ", subMasks)
    return subMasks

def calc_TVNorm(mask,p=3,q=3):
    '''
    Calculates the Total Variational Norm by summing the differences of the values in between the different positions 
    in the mask. p=3 and q=3 are defaults from the paper.
    '''
    val = 0
    for u in range(1,len(mask)-1):

        val += torch.abs(mask[u-1]-mask[u])**p
        val += torch.abs(mask[u+1]-mask[u])**p
    val = val**(1/p)
    val = val**q

    return val

def initMask(seq,model,intraBidx,target,thresh=0.9, mode="central", maskPertType='freeze'):
    '''
    Initiaizes the first value of the mask where the gradient descent methods for finding
    the masks starts. Central finds the smallest centered mask which still reduces the class score by 
    at least 90% compared to a fully perturbing mask (whole mask on). Random just turns (on average) 70% of the 
    mask (Does not give very conclusive results so far). 
    '''
    if(mode=="central"):
        
        #first define the fully perturbed sequence
        fullPert = torch.zeros_like(seq)
        for i in range(seq.shape[2]):
            fullPert[:,:,i,:,:] = seq[:,:,0,:,:]
        
        #get the class score for the fully perturbed sequence

        fullPertScore=model(fullPert)[intraBidx,target[intraBidx]]
        origScore = model(seq)[intraBidx,target[intraBidx]]
            
        mask = torch.cuda.FloatTensor([1]*seq.shape[2])
        
        #reduce mask size while the loss ratio remains above 90%
        for i in range(1,seq.shape[2]//2):
            newMask = torch.cuda.FloatTensor([1]*seq.shape[2])
            newMask[:i]=0
            newMask[-i:]=0

            centralScore=model(perturbSequence(seq,newMask,perbType=maskPertType))[intraBidx,target[intraBidx]]
            scoreRatio=(origScore-centralScore)/(origScore-fullPertScore)
            
            if(scoreRatio < thresh):
                break
            
        mask=newMask

        #modify the mask so that it is roughly 0 or 1 after sigmoid
        for j in range(len(mask)):
            if(mask[j]==0):
                mask[j]=-5
            elif(mask[j]==1):
                mask[j]=5
    
    elif(mode=="random"):
        #random init to 0 or 1, then modify for sigmoid
        mask = torch.cuda.FloatTensor(16).uniform_() > 0.7
        mask = mask.float()
        mask = mask - 0.5
        mask = mask*5
        
        #if mask were to be ALL 0's or 1's, perturb one a bit so that TV norm doesn't NaN
        if(torch.abs(mask.sum())==2.5*len(mask)):
            mask[8]+=0.1

    mask.requires_grad_()
    print("initial mask is: ", mask)
    return mask
      
def findMaskCombinatorial(seq,model,intraBidx,target,lam1,lam2, maskPertType='freeze', allowSplit=False, maxMaskSize=None):

    derp = 0
    if(allowSplit):
        return findMaskCombinatorialSplit(seq,model,intraBidx,target,lam1,lam2)
 
    if(maxMaskSize==None):
        maxMaskLen = range(1,seq.shape[2])
    else:
        maxMaskLen = range(1,maxMaskSize+1)
    bestMask = torch.zeros(seq.shape[2]).to(device)
    bestLoss= model(perturbSequence(seq,bestMask,perbType=maskPertType))[intraBidx,target[intraBidx]]

    sm = torch.nn.Softmax(dim=1)
    for m_size in maxMaskLen:
        for m_start in range((seq.shape[2]-m_size)+1):

            mask = torch.zeros(seq.shape[2]).to(device)
            mask[m_start:m_start+m_size]=1
            #curr_loss=sm(model(perturbSequence(seq,mask)))[intraBidx,target[intraBidx]]
            curr_loss=model(perturbSequence(seq,mask,perbType=maskPertType))[intraBidx,target[intraBidx]]
            l1loss = lam1*torch.sum(torch.abs(mask))
            tvnormLoss= lam2*calc_TVNorm(mask,p=3,q=3)
            curr_loss+=l1loss
            curr_loss+=tvnormLoss

            if(curr_loss<bestLoss):
                bestLoss=curr_loss
                bestMask=mask
    print("combinatorial mask is: ", bestMask)
    return bestMask

def findMaskCombinatorialSplit(seq,model,intraBidx,target,lam1,lam2, maskPertType='freeze'):

    bestMask = torch.zeros(seq.shape[2]).to(device)
    bestLoss= model(perturbSequence(seq,bestMask,perbType=maskPertType))[intraBidx,target[intraBidx]]
    #print("current loss is: ", bestLoss)
    sm = torch.nn.Softmax(dim=1)
    for m_size in range(1,5):
        for m_start in range(seq.shape[2]-m_size):            
            for m2_start in range(seq.shape[2]-m_size):
                mask = torch.zeros(seq.shape[2]).to(device)
                mask[m_start:m_start+m_size]=1
                mask[m2_start:m2_start+m_size]=1

                curr_loss=model(perturbSequence(seq,mask,perbType=maskPertType))[intraBidx,target[intraBidx]]
                l1loss = lam1*torch.sum(torch.abs(mask))
                tvnormLoss= lam2*calc_TVNorm(mask,p=3,q=3)
                curr_loss+=l1loss
                curr_loss+=tvnormLoss

                if(curr_loss<bestLoss):
                    bestLoss=curr_loss
                    bestMask=mask
    print("combinatorial mask is: ", bestMask)
    return bestMask

def vizualize_results(orig_seq, pert_seq, mask, rootDir=None,case="0",markImgs=True, iterTest=False):
    if(rootDir==None):
        rootDir="vizualisations/"+args.subDir+"/"
    #if(iterTest):
    rootDir+="/PerturbImgs/"
        
    if not os.path.exists(rootDir):
        os.makedirs(rootDir)
    
    origPy = orig_seq.cpu().detach().numpy()
    pertPy = pert_seq.cpu().detach().numpy()
    for i in range(orig_seq.shape[1]):

        if(markImgs):
            origPy[1:,i,:10,:10]=0
            origPy[0,i,:10,:10]=mask[i].detach().cpu()*255
            pertPy[1:,i,:10,:10]=0
            pertPy[0,i,:10,:10]=mask[i].detach().cpu()*255
        #result = Image.fromarray(origPy[:,i,:,:].transpose(1,2,0).astype(np.uint8))
        #result.save(rootDir+"case"+case+"orig"+str(i)+".png")
        result = Image.fromarray(pertPy[:,i,:,:].transpose(1,2,0).astype(np.uint8))
        result.save(rootDir+"case"+case+"pert"+str(i)+".png")
    f = open(rootDir+"case"+case+".txt","w+")
    f.write(str(mask.detach().cpu()))
    f.close()
    
def vizualize_results_on_gradcam(gradCamImage, mask, rootDir,case="0",roundUpMask=True, imageWidth=224, imageHeight=224):
        
    #print("gradCamType: ", gradCamImage.type)
    try:
        mask=mask.detach().cpu()
    except:
        print("mask was already on cpu")
    if not os.path.exists(rootDir):
        os.makedirs(rootDir)
    
    dots = findTempMaskRedDots(imageWidth,imageHeight,mask,roundUpMask)
    
    dotOffset = imageWidth*2
    for i in range(len(mask)):
        for j,dot in enumerate(dots):

            if(i==j):
                intensity=255
            else:
                intensity=150
            gradCamImage[:,i,dot["yStart"]:,dotOffset+dot["xStart"]:dotOffset+dot["xEnd"]]=0
            gradCamImage[dot["channel"],i,dot["yStart"]:,dotOffset+dot["xStart"]:dotOffset+dot["xEnd"]]=intensity

            result = Image.fromarray(gradCamImage[::-1,i,:,:].transpose(1,2,0).astype(np.uint8),mode="RGB")
            result.save(rootDir+"/case"+case+"_"+str(i)+".png")

    f = open(rootDir+"/MASKVALScase"+case+".txt","w+")
    f.write(str(mask.detach().cpu()))
    f.close()
    
def findTempMaskRedDots(imageWidth,imageHeight,mask,roundUpMask):
    maskLen = len(mask)
    dotWidth = int(imageWidth//(maskLen+4))
    dotPadding = int((imageWidth - (dotWidth*maskLen))//maskLen)
    dotHeight = int(imageHeight//20)
    dots = []
    
    for i,m in enumerate(mask):
        
        if(roundUpMask):
            if(mask[i]>0.5):
                mask[i]=1
            else:
                mask[i]=0
                
        dot={'yStart': -dotHeight,
             'yEnd' : imageHeight,
             'xStart' : i*(dotWidth+dotPadding),
             'xEnd' : i*(dotWidth+dotPadding)+dotWidth}
        if(mask[i]==0):
            dot['channel']=1
        else:
            dot['channel']=2
            
        dots.append(dot)
        
    return dots
        
def createImageArrays(input_sequence, gradcamMask,timeMask,intraBidx,temporalMaskType,output_folder,targTag,RESIZE_FLAG,RESIZE_SIZE_WIDTH,RESIZE_SIZE_HEIGHT):
    
    input_to_model = input_sequence[intraBidx][None,:,:,:,:]

    input_data_unnormalised = input_to_model[0].cpu().permute(1, 2, 3, 0).numpy() 
    input_data_unnormalised = np.flip(input_data_unnormalised, 3)
    
    combined_images = []
    for i in range(input_data_unnormalised.shape[0]):
        input_data_img = input_data_unnormalised[i, :, :, :]
        heatmap = cv2.applyColorMap(np.uint8(255 * gradcamMask[i]), cv2.COLORMAP_JET)
        if RESIZE_FLAG:
            input_data_img = cv2.resize(input_data_img, (RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT))
            heatmap = cv2.resize(heatmap, (RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT))
        heatmap = np.float32(heatmap)# / 255
        cam = heatmap + np.float32(input_data_img)
        cam = cam / np.max(cam)

        combined_img = np.concatenate((np.uint8(input_data_img), np.uint8(255* cam),\
                    np.uint8(perturbSequence(input_sequence,timeMask.clone(),\
                    perbType=temporalMaskType,snapValues=True)[intraBidx][:,i,:,:].permute(1,2,0).detach().cpu().numpy())[:,:,::-1])\
                                      , axis=1)#had 255* on both
        combined_images.append(combined_img)
        cv2.imwrite(os.path.join(output_folder, "img%02d.jpg" % (i + 1)), combined_img)

    path_to_combined_gif = os.path.join(output_folder, "mygif.gif")
    os.system("convert -delay 10 -loop 0 {}.jpg {}".format(
                                        os.path.join(output_folder, "*"),
                                        path_to_combined_gif))
    
    combined_images = np.transpose(np.array(combined_images), (3,0,1,2))
    vizualize_results_on_gradcam(combined_images, timeMask,rootDir=output_folder,case=temporalMaskType+targTag)
    
                    
    return combined_images
                    
def find_masks(dat_loader, model,hyper_params,lam1, lam2, N, ita, maskType="gradient", temporalMaskType="freeze", classOI=None, verbose=True, maxMaskLength=None, doGradCam=False, runTempMask=True):
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
    #classOI here is one of the 5 files we split up
    ##rebuttal stuff start ##
    rebuttalResultsPath = "ECCV_results/"
    clipsTimeMaskResults= []
    clipsGradCamResults = []
    if not os.path.exists(rebuttalResultsPath):
        os.makedirs(rebuttalResultsPath)
    ##rebuttal stuff end ##
    
    for i, (input, target, label) in enumerate(dat_loader):
        if(i%50==0):
            print("on idx: ", i)

        input_var = input.to(device)
        target = target.to(device)

        model.zero_grad()

        #eta is for breaking out of the grad desc early if it hasn't improved
        eta = 0.00001
        
        haveOutput=False
        
        for intraBidx in range(hyper_params["batch_size"]):

            
            #only look at cases where clip is of a certain class (if class of interest 'classoI' was given)
            currC = str(target[intraBidx].item())

            if (currC in list(df.keys()) and int(label[intraBidx]) in [clip for clip in df[currC]]) or classOI==None:                
                if not haveOutput:
                    output = model(input_var)
                    haveOutput=True

                if(runTempMask):
                    if(hyper_params["gradCamType"]=="guessed"):
                        maskTarget=torch.zeros((hyper_params["batch_size"],1)).long()
                        maskTarget[intraBidx]=torch.argmax(output[intraBidx])

                    else:
                        maskTarget=target
                #combinatorial mask finding is straight forward, needs no grad desc
                    if(maskType=="combi"):
                        timeMask = findMaskCombinatorial(input_var,model,intraBidx,maskTarget, lam1,lam2, maskPertType=temporalMaskType,maxMaskSize=maxMaskLength)
                    elif(maskType=="combis"):
                        timeMask = findMaskCombinatorial(input_var,model,intraBidx,maskTarget, lam1,lam2, maskPertType=temporalMaskType, maxMaskSize=maxMaskLength, allowSplit=True)

                    #gradient descent type
                    else:
                        model.zero_grad()
                        timeMask = initMask(input_var,model,intraBidx,maskTarget,thresh=0.9,mode="central", maskPertType=temporalMaskType)
                        optimizer = torch.optim.Adam([timeMask], lr=0.2)
                        oldLoss = 999999
                        for nidx in range(N):

                            if(nidx%25==0):
                                print("on nidx: ", nidx)

                            mask_clip = torch.sigmoid(timeMask)
                            l1loss = lam1*torch.sum(torch.abs(mask_clip))
                            tvnormLoss= lam2*calc_TVNorm(mask_clip,p=3,q=3)

                            classLoss = model(perturbSequence(input_var,mask_clip, perbType=temporalMaskType))

                            classLoss = classLoss[intraBidx,maskTarget[intraBidx]]

                            loss = l1loss + tvnormLoss + classLoss

                            if(abs(oldLoss-loss)<eta):
                                break;

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        timeMask = torch.sigmoid(timeMask)
                        scoreSavePath=os.path.join("cam_saved_images",args.subDir,str(target[intraBidx].item()),\
                                                   label[intraBidx]+"g_"+str(torch.argmax(output[intraBidx]).item())\
                                                   +"_gs%5.4f"%torch.max(output[intraBidx]).item()+"_cs%5.4f"%output[intraBidx][target[intraBidx]].item(), "combined")

                        if not os.path.exists(scoreSavePath):
                            os.makedirs(scoreSavePath)

                        f = open(scoreSavePath+"/ClassScoreFreezecase"+label[intraBidx]+".txt","w+")
                        f.write(str(classLoss.item()))
                        f.close()

                        classLossFreeze = model(perturbSequence(input_var,timeMask, perbType="reverse"))
                        classLossFreeze = classLossFreeze[intraBidx,maskTarget[intraBidx]]

                        f = open(scoreSavePath+"/ClassScoreReversecase"+label[intraBidx]+".txt","w+")
                        f.write(str(classLossFreeze.item()))
                        f.close()
                        
                        #as soon as you have the time mask, and freeze/reverse scores,
                        #Add results for current clip in list of timemask results
                        clipsTimeMaskResults.append({'true_class' : int(target[intraBidx].item()),
                                              'pred_class' : int(torch.argmax(output[intraBidx]).item()),
                                              'video_id' : int(label[intraBidx]),
                                              'time_mask' : timeMask.detach().cpu().numpy(),
                                              'original_score_guess' : torch.max(output[intraBidx]).item(),
                                              'original_score_true' : output[intraBidx][target[intraBidx]].item(),
                                              'freeze_score' : classLoss.item(),
                                              'reverse_score' : classLossFreeze.item()
                                             })

                if(doGradCam):  

                    target_index = maskTarget[intraBidx]

                    RESIZE_SIZE_WIDTH = 224
                    RESIZE_SIZE_HEIGHT = 224

                    grad_cam = GradCamVideo(model=model.module,
                                       target_layer_names=['Mixed_5c'],#model.module.end_points and ["block5"],
                                       class_dict=None,
                                       use_cuda=True,
                                       input_spatial_size=(RESIZE_SIZE_WIDTH,RESIZE_SIZE_HEIGHT),
                                       normalizePerFrame=True,
                                       archType="I3D")
                    input_to_model = input_var[intraBidx][None,:,:,:,:]

                    if(hyper_params["gradCamType"]=="guessed"):
                        target_index=torch.argmax(output[intraBidx])

                    mask, output_grad = grad_cam(input_to_model, target_index)

                    #as soon as you have the numpy array of gradcam heatmap, add it to list of GCheatmaps
                    clipsGradCamResults.append({'true_class' : int(target[intraBidx].item()),
                                          'pred_class' : int(torch.argmax(output[intraBidx]).item()),
                                          'video_id' : int(label[intraBidx]),
                                          'GCHeatMap' : mask
                                         })

                    '''beginning of gradcam write to disk'''
                    input_data_unnormalised = input_to_model[0].cpu().permute(1, 2, 3, 0).numpy() 
                    input_data_unnormalised = np.flip(input_data_unnormalised, 3)

                    targTag=label[intraBidx]

                    output_images_folder_cam_combined = os.path.join("cam_saved_images",args.subDir,str(target[intraBidx].item()),targTag+"g_"+str(torch.argmax(output[intraBidx]).item())+"_gs%5.4f"%torch.max(output[intraBidx]).item()+"_cs%5.4f"%output[intraBidx][target[intraBidx]].item(), "combined")

                    os.makedirs(output_images_folder_cam_combined, exist_ok=True)

                    clip_size = mask.shape[0]

                    RESIZE_FLAG = 0
                    SAVE_INDIVIDUALS = 1

                if(doGradCam and runTempMask):
                    createImageArrays(input_var, mask,timeMask,intraBidx,"freeze",output_images_folder_cam_combined,targTag,\
                                     RESIZE_FLAG,RESIZE_SIZE_WIDTH,RESIZE_SIZE_HEIGHT)
                    createImageArrays(input_var, mask,timeMask,intraBidx,"reverse",output_images_folder_cam_combined,targTag,\
                                     RESIZE_FLAG,RESIZE_SIZE_WIDTH,RESIZE_SIZE_HEIGHT)

                    masks.append(timeMask)
                    
    #finally, write pickle files to disk
    f = open(rebuttalResultsPath+"allTimeMaskResults_"+args.subDir+"_"+classOI+"_"+".p","wb")
    pickle.dump(clipsTimeMaskResults, f)
    f.close()
    
    f = open(rebuttalResultsPath+"allGradCamResults_"+args.subDir+"_"+classOI+"_"+".p","wb")
    pickle.dump(clipsGradCamResults, f)
    f.close()
    
    return masks




if __name__ == '__main__':
    main()
