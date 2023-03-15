import os
import csv
import argparse

import torch
from tqdm import tqdm

import data_util
import model_util
import attack_util
from attack_util import ctx_noparamgrad

from matplotlib import pyplot as plt

def get_model(norm_layer):
    model = model_util.ResNet18(num_classes=10)
    model.normalizer = norm_layer
    return model

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eps", type=int, default=8, help="Attack budget: epsilon / 255"
    )
    parser.add_argument(
        "--alpha", type=float, default=2, help="PGD attack step size: alpha / 255"
    )
    parser.add_argument(
        "--attack_step", type=int, default=10, help="Number of PGD iterations"
    )
    parser.add_argument(
        "--loss_type", type=str, default="ce", choices=['ce', 'cw'], help="Loss type for attack"
    )
    parser.add_argument(
        '--data_dir', default='./data/', type=str, help="Folder to store downloaded dataset"
    )
    parser.add_argument(
        '--model_path', default='resnet_cifar10.pth', help='Filepath to the trained model'
    )
    parser.add_argument("--targeted", action='store_true')
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()
    return args


def plot_accuracy(standard_acc, standard_vars, robust_acc, x_label, colors=['orange', 'green']):
    plt.plot(standard_vars, standard_acc, label='Standard Model', color=colors[0], marker='o')
    plt.plot(standard_vars, robust_acc, label='Robust Model', color=colors[1], marker='o')

    plt.title(f'Standard and Robust Accuracies by different {x_label}')
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def main():
    args = parse_args()

    # Get model and dataset
    train_loader, val_loader, test_loader, norm_layer = data_util.cifar10_dataloader(data_dir=args.data_dir)
    model = get_model(norm_layer)
    #args.model_path = "./checkpoints/checkpoint_100.pth"
    if args.model_path != "":
       model.load(args.model_path, 'cpu')

    model = model.to(args.device)

    eps = args.eps / 255
    alpha = args.alpha / 255

    attacker = attack_util.PGDAttack(
        attack_step=args.attack_step, eps=eps, alpha=alpha, loss_type=args.loss_type,
        targeted=args.targeted, num_classes=10)

    total = 0
    clean_correct_num = 0
    robust_correct_num = 0
    target_label = 1  ## only for targeted attack

    ## Make sure the model is in `eval` mode.
    model.eval()
    sample_no = 0

    for data, labels in tqdm(test_loader):
        sample_no += 1

        data = data.to(args.device)
        labels = labels.to(args.device)
        if args.targeted:
            data_mask = (labels != target_label)
            if data_mask.sum() == 0:
                continue
            data = data[data_mask]
            labels = labels[data_mask]
            attack_labels = torch.ones_like(labels).to(args.device)
        else:
            attack_labels = labels
        attack_labels = attack_labels.to(args.device)
        batch_size = data.size(0)
        total += batch_size

        with ctx_noparamgrad(model):
            ### clean accuracy
            predictions = model(data)
            clean_correct_num += torch.sum(torch.argmax(predictions, dim=-1) == labels).item()

            ### robust accuracy
            # generate perturbation
            perturbed_data = attacker.perturb(model, data, attack_labels) + data

            # predict
            predictions = model(perturbed_data)
            robust_correct_num += torch.sum(torch.argmax(predictions, dim=-1) == labels).item()

    print(
        f"Total number of images: {total}\nClean accuracy: {clean_correct_num / total}\nRobust accuracy {robust_correct_num / total}")


if __name__ == "__main__":
    main()
