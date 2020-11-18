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
import loss_functions
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
img_shape = (3, 480, 640) #TODO: img shape
debug = False 

def main():
    global best_error, n_iter, device, scheduler
    args = parser.parse_args()

    print('=> will save everything to {}'.format(args.save_path))
    if debug:
        st()
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if args.evaluate:
        #TODO load model 
        args.epochs = 0
        _, val_loader = create_data_loaders(args)
        checkpoint = torch.load(args.pretrained_net)
        model.load_state_dict(checkpoint['model_state_dict'])
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return
    else:
        tb_writer = SummaryWriter(args.save_path)
        train_loader, val_loader = create_data_loaders(args)
        if args.epoch_size == 0:
            args.epoch_size = len(train_loader)
        # create model
        print("=> creating models")
        if debug:
            st()
        model = models.Runner(args)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)
        model = model.cuda()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15000, gamma=0.5)
   
        logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
        logger.epoch_bar.start()

        print("Training...")
        if debug:
            st()
        for epoch in range(args.epochs):
            logger.epoch_bar.update(epoch)

            # train for one epoch
            logger.reset_train_bar()
            train_loss = train(args, train_loader, model, optimizer, scheduler, args.epoch_size, logger, tb_writer)
            logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

            # evaluate on validation set
            logger.reset_valid_bar()
            if epoch%1==0: # TODO: fix this 
                print("Starting Validation")
                if debug:
                    st()
                # if (epoch+2)%4==0:
                errors, error_names = validate(args, val_loader_gt, lstmNet, epoch, logger, tb_writer)
            
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
            save_checkpoint(model.state_dict(),is_best, epoch, args.save_path)

        logger.epoch_bar.finish()


def train(args, train_loader, model, optimizer, scheduler, epoch_size, logger, tb_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 ,w4 = args.appearance_loss_weight, args.smooth_loss_weight, args.trajectory_loss_weight, args.gan_loss_weight

    model.train()

    end = time.time()
    logger.train_bar.update(0)
  
    for i, (input, target) in enumerate(train_loader):
        # st()
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        loss = model(tgt_img, ref_imgs, intrinsics, log_losses, log_output, tb_writer, n_iter, False,\
                                                          mode ='train')
        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

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

def validate (val_loader, model, epoch, write_to_file=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    #TODO comlete it
    


if __name__ == '__main__':
    main()