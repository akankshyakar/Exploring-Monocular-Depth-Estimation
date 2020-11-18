import argparse
import time
import csv
import os
import torch
from torch.autograd import Variable
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
import models
import utils
from utils import tensor2array, save_checkpoint, save_path_formatter, log_output_tensorboard
import pdb
from loss_functions import photometric_reconstruction_loss, explainability_loss, smooth_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
import torchvision
import ipdb 
st=  ipdb.set_trace
torch.manual_seed(125)
torch.cuda.manual_seed_all(125) 
torch.backends.cudnn.deterministic = True

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
adversarial_loss = torch.nn.BCELoss()
img_shape = (3, 128, 416) #TODO img shaoe
debug = False 

def main():
    global best_error, n_iter, device, scheduler
    args = parser.parse_args()
    if args.use_batchnorm:
        print("Will use batchnorm")

    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from datasets.sequence_folders import SequenceFolder
    save_path = save_path_formatter(args, parser)
    args.save_path = os.path.join(args.log_dir,save_path)
    print('=> will save everything to {}'.format(args.save_path))
    if debug:
        st()
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if args.evaluate:
        args.epochs = 0

    tb_writer = SummaryWriter(args.save_path)
    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        # torchvision.transforms.Resize((128,416)),
        custom_transforms.RandomHorizontalFlip(),
        # custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    if debug:
        st()
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    #if args.with_gt:
    from datasets.validation_seq_folders import ValidationSet
    from datasets.train_seq_folders import TrainValidationSet
    full_seq = False
    if args.pretrained_net:
        full_seq = True

    val_set_gt = ValidationSet(
        args.data,
        transform=valid_transform,
        full_seq = full_seq
    )
    if args.eval_train:
        print("Evaluating on training dataset")
        val_set_gt = TrainValidationSet(
            args.data,
            transform=valid_transform,
            full_seq = full_seq
        )
    #else:
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
    )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    if debug:
        st()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_loader_gt = torch.utils.data.DataLoader(
        val_set_gt, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating models")
    if debug:
        st()
    lstmNet = models.LSTMRunner(args)
    output_exp = args.appearance_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
    cudnn.benchmark = True
    print('=> setting solvers')
    if debug:
        st()

    if args.discriminator_type == "Encoder": #SGD for encoder discriminator
        optim_params_disc = [{'params': lstmNet.discriminator.parameters(), 'lr':args.lr}]
        optimizer_disc = torch.optim.SGD(optim_params_disc,weight_decay=args.weight_decay)
        scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_disc, 15000, gamma=0.5)
    else:
        optim_params_disc = [{'params': lstmNet.discriminator.parameters(), 'lr': args.lr}]
        optimizer_disc = torch.optim.Adam(optim_params_disc,betas=(args.momentum, args.beta),weight_decay=args.weight_decay)
        scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_disc, 15000, gamma=0.5)
    #adam for pose and disp and code
    optim_params_others = [{'params': list(lstmNet.lstm.parameters()) + list(lstmNet.disp_net.parameters()) + list(lstmNet.pose_exp_net.parameters()) + list(lstmNet.encoder.parameters()), 'lr':args.lr}]
    optimizer_others = torch.optim.Adam(optim_params_others,betas=(args.momentum, args.beta),weight_decay=args.weight_decay)
    scheduler_others = torch.optim.lr_scheduler.StepLR(optimizer_others, 15000, gamma=0.5)

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    if args.pretrained_net or args.evaluate:
        logger.reset_valid_bar()
        checkpoint = torch.load(args.pretrained_net)
        lstmNet.load_state_dict(checkpoint['model_state_dict'])
        errors, error_names = validate_with_gt(args, val_loader_gt, lstmNet, 0, logger, tb_writer)
        '''for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, 0)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[2:9], errors[2:9]))
        logger.valid_writer.write(' * Avg {}'.format(error_string))'''
        for val, name in zip(errors, error_names):
            print(name, ":", val)
        return

    print("Training...")
    if debug:
        st()
    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, lstmNet, optimizer_disc, optimizer_others, args.epoch_size, logger, tb_writer, scheduler_disc, scheduler_others)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        if epoch%1==0: # TODO: fix this for lstm code
            print("Starting Validation")
            if debug:
                st()
            # if (epoch+2)%4==0:
            errors, error_names = validate_with_gt(args, val_loader_gt, lstmNet, epoch, logger, tb_writer)
            # else:
            #     errors, error_names = validate_without_gt(args, val_loader, lstmNet, epoch, logger, tb_writer)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(args.save_path, lstmNet.state_dict(),is_best, epoch, args.expname + '_disc', optimizer_disc)
        save_checkpoint(args.save_path, lstmNet.state_dict(),is_best, epoch, args.expname + '_others', optimizer_others)

    logger.epoch_bar.finish()


def train(args, train_loader, lstmNet, optimizer_disc, optimizer_others, epoch_size, logger, tb_writer, scheduler_disc, scheduler_others):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 ,w4 = args.appearance_loss_weight, args.smooth_loss_weight, args.trajectory_loss_weight, args.gan_loss_weight

    lstmNet.train()

    end = time.time()
    logger.train_bar.update(0)
  
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        # st()
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        # pdb.set_trace()
        intrinsics = intrinsics.to(device)

        loss, loss_1, loss_2, loss_3, gan_loss = lstmNet(tgt_img, ref_imgs, intrinsics, log_losses, log_output, tb_writer, n_iter, False,\
                                                          mode ='train')
        # pdb.set_trace()
        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do step

        optimizer_others.zero_grad()
        optimizer_disc.zero_grad()
        loss.backward()
        optimizer_others.step()
        scheduler_others.step()
        optimizer_disc.step()
        scheduler_disc.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


if __name__ == '__main__':
    main()
