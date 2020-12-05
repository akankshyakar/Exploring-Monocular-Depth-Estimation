import argparse
import time
import csv
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from torch.autograd import Variable
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import custom_transforms
import models
import utils
from utils import save_checkpoint, create_data_loaders, parse_command, compute_depth_metrics, tensor2array
import pdb
st = pdb.set_trace
# import loss_functions
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
import torchvision
from path import Path
torch.manual_seed(125)
torch.cuda.manual_seed_all(125) 
torch.backends.cudnn.deterministic = True

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
debug = False 
args = utils.parse_command()

def main():
    global best_error, n_iter, device, scheduler
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print("torch.cuda.is_available()", torch.cuda.is_available())
    # st()
    print('=> will save everything to {}'.format(args.save_path))
    save_path = Path(args.save_path)
    save_path.makedirs_p()
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
        model = models.Runner(args)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)
        model = model.cuda()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 16000, gamma=0.5)
   
        logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
        logger.epoch_bar.start()

        print("Training...")
        if debug:
            st()
        for epoch in range(args.epochs+1):
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
                errors, error_names = validate(args, val_loader, model, epoch, logger, tb_writer)
            
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
    model.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, data in enumerate(train_loader):
        # if i >=463:
        #     st()
        if args.data == 'nyudepthv2':
            (img, ref_imgs, depth, intrinsics, mask_gt, world_coords_gt) = data
            depth = depth.float().to(device)
            img = img.float().to(device)
            ref_imgs = [img.float().to(device) for img in ref_imgs]
            intrinsics = intrinsics.float().to(device)
            # TODO(abhorask) throw error if these dummy values are used
            mask_gt = mask_gt.to(device)
            world_coords_gt = world_coords_gt.to(device)
        elif args.data == 'sunrgbd':
            img, depth, mask_gt, intrinsics, world_coords_gt = data['image'], data['depth'], data['mask'], data['intrinsics'], data['world_pcl']
            depth = depth.unsqueeze(1).float().to(device)
            img = img.float().to(device)
            ref_imgs = []
            intrinsics = intrinsics.float().to(device)
            mask_gt = mask_gt.float().to(device)
            world_coords_gt = world_coords_gt.float().to(device)

        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)


        # st()
        loss = model(img, ref_imgs, intrinsics, depth, mask_gt, world_coords_gt, log_losses, log_output, tb_writer, n_iter, False, mode ='train', args=args)
        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do step
        optimizer.zero_grad()
        # import pdb; pdb.set_trace()
        # print("train 124 loss", loss)
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

def validate (args, val_loader, model, epoch, logger, tb_writer, log_outputs=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_output = 1 > 0
    #TODO complete it
    depth_metric_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    depth_errors = AverageMeter(i=len(depth_metric_names))
    #TODO pose
    model.eval()
    end = time.time()
    logger.valid_bar.update(0)
    for i, data in enumerate(val_loader):

        if args.data == 'nyudepthv2':
            (img, ref_imgs, gt_depth, intrinsics, mask_gt, world_coords_gt) = data
            gt_depth = gt_depth.to(device)
            img = img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]
            intrinsics = intrinsics.to(device)
            # TODO(abhorask) throw error if these dummy values are used
            mask_gt = mask_gt.to(device)
            world_coords_gt = world_coords_gt.to(device)
        elif args.data == 'sunrgbd':
            img, gt_depth, mask_gt, intrinsics, world_coords_gt = data['image'], data['depth'], data['mask'], data['intrinsics'], data['world_pcl']
            gt_depth = gt_depth.unsqueeze(1).to(device)
            img = img.to(device)
            ref_imgs = []
            intrinsics = intrinsics.to(device)
            mask_gt = mask_gt.to(device)
            world_coords_gt = world_coords_gt.to(device)

        output_depth=model(img, ref_imgs, intrinsics, gt_depth, mask_gt, world_coords_gt, False, log_output, tb_writer, n_iter, ret_depth=True, mode ='val', args=args)
        output_disp = 1/output_depth[0]
        if log_outputs and i < 3:
            if epoch == 0:
                tb_writer.add_image('val Input/{}'.format(i), tensor2array(img[0]), 0)
                depth_to_show = gt_depth[0]
                tb_writer.add_image('val target Depth Normalized/{}'.format(i),
                                    tensor2array(depth_to_show, max_value=None),
                                    epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0, 10)
                tb_writer.add_image('val target Disparity Normalized/{}'.format(i),
                                    tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                    epoch)

            tb_writer.add_image('val Dispnet Output Normalized/{}'.format(i),
                                tensor2array(output_disp[0], max_value=None, colormap='magma'),
                                epoch)
            tb_writer.add_image('val Depth Output Normalized/{}'.format(i),
                                tensor2array(output_depth[0], max_value=None),
                                epoch)
        # st()
        depth_errors.update(compute_depth_metrics(gt_depth, output_depth[0]))

        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, depth_errors.val[0], depth_errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
    return depth_errors.avg, depth_metric_names

if __name__ == '__main__':
    main()
