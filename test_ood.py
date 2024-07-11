from collections import OrderedDict
from mobilenet import mobilenet_v2
from resnet import resnet50
from wrn import WideResNet
from resnetv2 import ResNet50
import torchvision as tv
from torchvision import transforms
import torch
import numpy as np
from utils.test_utils import arg_parser, get_measures
from utils import log
import pathlib
import torch.nn as nn
import os
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import time
import multiprocessing

num_workers = min(4, multiprocessing.cpu_count())


def make_id_ood(args, logger):
    """Returns train and validation datasets."""
    if args.in_datadir == 'cifar10':
        # mean and standard deviation of channels of CIFAR-10 images
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        crop = 32
        val_tx = tv.transforms.Compose([
            tv.transforms.Resize((crop, crop)), tv.transforms.ToTensor(), tv.transforms.Normalize(mean, std)])
        in_set = tv.datasets.CIFAR10('./dataset/id_data/cifarpy', train=False,  
                                     download=True, transform=val_tx)

    elif args.in_datadir == 'cifar100':
        # mean and standard deviation of channels of CIFAR-100 images
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        crop = 32
        val_tx = tv.transforms.Compose([
            tv.transforms.Resize((crop, crop)), tv.transforms.ToTensor(), tv.transforms.Normalize(mean, std)])
        in_set = tv.datasets.CIFAR100('./dataset/id_data/cifarpy', train=False,
                                      download=True, transform=val_tx)

    elif args.in_datadir == 'imageNet':
        print('dataset: imageNet')
        val_tx = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        in_set = tv.datasets.ImageFolder('./dataset/id_data/imageNet_dataset/',
                                         transform=val_tx)
    else:
        crop = 480
        val_tx = tv.transforms.Compose([
            tv.transforms.Resize((crop, crop)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)

    if 'SVHN' in args.out_datadir:
        out_set = tv.datasets.SVHN(root='./dataset/ood_data/svhn', split="test", download=True, transform=val_tx)
    elif pathlib.Path(args.out_datadir).name == 'CIFAR10':
        out_set = tv.datasets.CIFAR10('./dataset/ood_data/cifarpy', train=False, download=True, transform=val_tx)
    elif pathlib.Path(args.out_datadir).name == 'CIFAR100':
        out_set = tv.datasets.CIFAR100('./dataset/ood_data/cifarpy', train=False, download=True, transform=val_tx)
    else:
        out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader

def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)
            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs = model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))

    return np.array(confs)


def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)
            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def iterate_data_maxlogit(data_loader, model):
    confs = []

    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)
            logits = logits.data.cpu().numpy()
            confs.extend([np.max(logits[i]) for i in range(logits.shape[0])])

    return np.array(confs)

def run_eval(model, in_loader, out_loader, logger, args, num_classes):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    if args.score == 'MSP':
        args.score = 'MSP'
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)
    elif args.score == 'ODIN':
        args.score = 'ODIN'
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    elif args.score == 'Energy':
        args.score = 'Energy'
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)

        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'Max-logit':
        args.score = 'Max-logit'
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_maxlogit(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_maxlogit(out_loader, model)
    else:
        raise ValueError("Unknown score type {}".format(args.score))


    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    logger.info('============Results for {}============'.format(args.score))
    logger.info('FPR95: {}'.format(round(fpr95 * 100, 2)))
    logger.info('AUROC: {}'.format(round(auroc*100, 2)))
    logger.info('AUPR (In): {}'.format(round(aupr_in*100, 2)))
    logger.info('AUPR (Out): {}'.format(round(aupr_out*100, 2)))

def main(args):
    logger = log.setup_logger(args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.backends.cudnn.benchmark = True

    in_set, out_set, in_loader, out_loader = make_id_ood(args, logger)

    logger.info(f"Loading model from {args.model_path}")

    if 'cifar10_resnet50' in args.model_path or 'cifar100_resnet50' in args.model_path:
        model = ResNet50(num_classes=100 if '100' in args.model_path else 10)
        checkpoint = torch.load(args.model_path, map_location=torch.device(device))
        state_dict = checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module' prefix from variable names.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f"ResNet50 model best accuracy: {checkpoint['acc']}")
    elif 'cifar10_wrn' in args.model_path or 'cifar100_wrn' in args.model_path:
        model = WideResNet(40, len(in_set.classes), 2, dropRate=0.3)
        model.load_state_dict(torch.load(args.model_path))
        print("WideResNet model loaded")

    elif 'imagenet_mobilenet' in args.model_path:
        ## MobileNet-v2
        model = mobilenet_v2(num_classes=1000, pretrained=True)
    elif 'imagenet_resnet' in args.model_path:
        ## resnet50 
        model = resnet50(num_classes=1000, pretrained=True)

    else:
        raise ValueError("Unknown model type or path: {}".format(args.model_path))

    model = model.cuda()

    start_time = time.time()
    run_eval(model, in_loader, out_loader, logger, args, num_classes=len(in_set.classes))
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Max-logit'], default='Max-logit')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0014, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    main(parser.parse_args())

