import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import numpy as np
from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import datasets_video
from scipy.stats import pearsonr

def ccc(y_true, y_pred):
    true_mean = torch.mean(y_true)
    pred_mean = torch.mean(y_pred)
    v_pred = y_pred - pred_mean
    v_true = y_true - true_mean
    
    rho =  torch.sum(v_pred*v_true) / (np.sqrt(torch.sum(v_pred**2)) * np.sqrt(torch.sum(v_true**2)))
    std_predictions = torch.std(y_pred)
    std_gt = torch.std(y_true)
    
    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)
    return ccc, rho   
best_loss = 1000

def main():
    global args, best_loss
    args = parser.parse_args()
    check_rootfolders()

    root_data, train_dict, val_dict = datasets_video.return_dataset(args.dataset, args.modality, args.view)
    num_class = 1


    args.store_name = '_'.join(['TRN', args.label_name, args.dataset, args.modality, args.view, args.arch, args.consensus_type, 'segment%d'% args.num_segments])
    print('storing name: ' + args.store_name)
    img_tmpl = '{:06d}.jpg' if args.view=='body' else 'frame_det_00_{:06d}.bmp'
    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                fix_all_weights = args.fix_all_weights)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation() # augmentation increase 

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(root_data, args.train_dict, args.label_name, num_segments=args.num_segments, phase='Train',
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=img_tmpl,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(root_data, args.val_dict, args.label_name, num_segments=args.num_segments, phase='Validation',
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=img_tmpl,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll': 
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif args.loss_type == 'mse':
        criterion = torch.nn.MSELoss().cuda()
    elif args.loss_type == 'mae':
        criterion = torch.nn.SmoothL1Loss().cuda()
    else: # another loss is mse or mae
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        log_val = open(os.path.join(args.root_log, '%s_val.csv' % args.store_name), 'w')
        validate(val_loader, model, criterion, 0, log_val)
        return

    log_training = open(os.path.join(args.root_log, '%s.csv' % args.store_name), 'w')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training)

            # remember best prec@1 and save checkpoint
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    CCCs = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type('torch.FloatTensor').cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        train_ccc = ccc(target, output.squeeze().data)[0]
        CCCs.update(train_ccc)
        losses.update(loss.data[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'CCC {CCCs.val: .4f} ({CCCs.avg: .4f})\t'
                    .format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr'], CCCs = CCCs))
            print(output)
            log.write(output + '\n')
            log.flush()



def validate(val_loader, model, criterion, iter, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    CCCs = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type('torch.FloatTensor').cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        val_ccc = ccc(target, output.squeeze().data)[0]
        CCCs.update(val_ccc)
        losses.update(loss.data[0], input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CCC {CCCs.val: .4f} ({CCCs.avg: .4f})\t'
                  .format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, CCCs = CCCs))
            print(output)
            log.write(output + '\n')
            log.flush()

    output = ('Testing Results: Loss {loss.avg:.5f} CCC {CCCs.avg: .5f}'
          .format( loss=losses, CCCs = CCCs))
    print(output)
    log.write(output +  '\n')
    log.flush()

    return loss.data[0]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),'%s/%s_best.pth.tar' % (args.root_model, args.store_name))

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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


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

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
