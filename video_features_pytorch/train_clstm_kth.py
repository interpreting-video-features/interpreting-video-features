import os
import sys
import time
import torch
import signal
import importlib
import numpy as np
import utils
from visualisation import PlotLearning

# load configurations
args = utils.load_args()
config = utils.load_module(args.config).config


# set column model
file_name = config['conv_model']
cnn_def = importlib.import_module('{}'.format(file_name))

# setup device - CPU or GPU
device, device_ids = utils.setup_cuda_devices(args)
print(' > Using device: {}'.format(device.type))
print(' > Active GPU ids: {}'.format(device_ids))

best_loss = float('Inf')

if config['input_mode'] == 'jpg':
    from data_loader_kth import KTHImLoader
else:
    raise ValueError('Please provide a valid input mode')


def main():
    global args, best_loss

    # set run output folder
    config['model_name'] = config['model_name'] + str(args.iteration)
    model_name = config['model_name']
    output_dir = config['output_dir']
    save_dir = os.path.join(output_dir, model_name)
    print(' > Output folder for this run -- {}'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))

    # assign Ctrl+C signal handler
    signal.signal(signal.SIGINT, utils.ExperimentalRunCleaner(save_dir))

    # create model
    print(' > Creating model ... !')
    model = cnn_def.Model(num_classes=6, nb_lstm_units=config['clstm_hidden'], channels=3,
                          conv_kernel_size=(5, 5), top_layer=True, avg_pool=False,
                          batch_normalization=config['batch_norm'], lstm_layers=config['clstm_layers'],
                          step=32, dropout=config['dropout'], conv_stride=config['conv_stride'],
                          image_size=(160, 120), effective_step=[7, 15, 23, 31])

    # multi GPU setting
    # model = torch.nn.DataParallel(model, device_ids).to(device)

    # optionally resume from a checkpoint
    checkpoint_path = config['pretrained_model_path']

    if args.resume:
        if os.path.isfile(checkpoint_path):
            print(" > Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(checkpoint_path)
            init_dict = model.module.state_dict()
            # pretrained_dict = {k: v for k,v in checkpoint.items() if k in init_dict}
            # print('pretrain keys are: ', list(pretrained_dict.keys()))
            pretrained_dict = {}

            try:
                args.start_epoch = checkpoint['epoch']
            except:
                args.start_epoch = 0

            try:
                best_loss = checkpoint['best_loss']
            except:
                best_loss = 100

            for k, v in checkpoint.items():
                if k != 'logits.conv3d.weight' and k != 'logits.conv3d.bias':
                    pretrained_dict[k] = v
                else:
                    print('blocked class layer!')

            init_dict.update(pretrained_dict)
            model.module.load_state_dict(init_dict)
            print(" > Loaded checkpoint '{}'"
              .format(checkpoint_path))
    else:
        print(" !#! No checkpoint found at '{}'".format(
            checkpoint_path))

    # define augmentation pipeline
    upscale_size_train = int(config['input_spatial_size'] * config['upscale_factor_train'])
    upscale_size_eval = int(config['input_spatial_size'] * config['upscale_factor_eval'])

    train_data = KTHImLoader(config['data_folder'] + 'train', clip_size=32)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=config['shuffle'],
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)

    val_data = KTHImLoader(config['data_folder'] + 'test', clip_size=32)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config['batch_size'], shuffle=config['shuffle'],
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)

    test_data = KTHImLoader(config['data_folder'] + 'test', clip_size=32)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)

    # print(' > Number of dataset classes : {}'.format(len(train_data.classes)))
    # assert len(train_data.classes) == hyper_params['num_classes']

    # define loss function (criterion)
    if config['soft_max']:
        criterion = torch.nn.NLLLoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    # define optimizer
    lr = config['lr']
    last_lr = config['last_lr']
    momentum = config['momentum']
    weight_decay = config['weight_decay']

    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                    momentum=config['momentum'],
                                    weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                     weight_decay=config['weight_decay'])
    if args.eval_only:
        validate(val_loader, model, criterion, train_data.classes_dict)
        print(' > Evaluation DONE !')
        return

    # set callbacks
    plotter = PlotLearning(os.path.join(
        save_dir, 'plots'), config['num_classes'])
    lr_decayer = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=2, verbose=True)
    val_loss = float('Inf')

    # set end condition by num epochs
    num_epochs = int(config['num_epochs'])
    if num_epochs == -1:
        num_epochs = 999999

    print(' > Training is getting started...')
    print(' > Training takes {} epochs.'.format(num_epochs))
    start_epoch = args.start_epoch if args.resume else 0

    for epoch in range(start_epoch, num_epochs):

        lrs = [params['lr'] for params in optimizer.param_groups]
        print(' > Current LR(s) -- {}'.format(lrs))
        if np.max(lr) < last_lr and last_lr > 0:
            print(' > Training is DONE by learning rate {}'.format(last_lr))
            sys.exit(1)

        # train for one epoch
        train_loss, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)

        # set learning rate
        lr_decayer.step(val_loss, epoch)

        # plot learning
        plotter_dict = {}
        plotter_dict['loss'] = train_loss
        plotter_dict['val_loss'] = val_loss
        plotter_dict['acc'] = train_top1 / 100
        plotter_dict['val_acc'] = val_top1 / 100
        plotter_dict['learning_rate'] = lr
        plotter.plot(plotter_dict)

        print(' > Validation loss after epoch {} = {}'.format(epoch, val_loss))

        # remember best loss and save the checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'Conv4Col',
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best, config)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # if(i==10):
        #    break
        # measure data loading time
        data_time.update(time.time() - end)

        if config['nclips_train'] > 1:
            input_var = list(input.split(config['clip_size'], 2))
            for idx, inp in enumerate(input_var):
                input_var[idx] = inp.to(device)
        else:
            input_var = input.to(device)

        target = target.to(device)

        model.zero_grad()

        # compute output and loss
        output = model(input_var)
        # print('output shape:', output.shape)
        # print('target shaep:', target.shape)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, class_to_idx=None):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to evaluate mode
    model.train()

    logits_matrix = []
    features_matrix = []
    targets_list = []
    item_id_list = []

    end = time.time()
    # with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):

        if config['nclips_val'] > 1:
            input_var = list(input.split(config['clip_size'], 2))
            for idx, inp in enumerate(input_var):
                input_var[idx] = inp.to(device)
        else:
            input_var = input.to(device)

        target = target.to(device)

        # compute output and loss
        output = model(input_var)
        loss = criterion(output, target)

        if args.eval_only:
            logits_matrix.append(output.cpu().data.numpy())
            # features_matrix.append(features.cpu().data.numpy())
            targets_list.append(target.cpu().numpy())
            # item_id_list.append(item_id)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['print_freq'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    """
    if args.eval_only:
        logits_matrix = np.concatenate(logits_matrix)
        features_matrix = np.concatenate(features_matrix)
        targets_list = np.concatenate(targets_list)
        item_id_list = np.concatenate(item_id_list)
        print(logits_matrix.shape, targets_list.shape, item_id_list.shape)
        save_results(logits_matrix, features_matrix, targets_list,
                     item_id_list, class_to_idx, config)
        get_submission(logits_matrix, item_id_list, class_to_idx, config)
    """
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()