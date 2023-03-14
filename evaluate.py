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

    # Load data
    test_loader, norm_layer = data_util.cifar10_dataloader(data_dir=args.data_dir)
    num_classes = 10
    model = model_util.ResNet18(num_classes=num_classes)
    model.normalize = norm_layer
    model.load(args.model_path, args.device)
    model = model.to(args.device)

    eps = args.eps / 255
    alpha = args.alpha / 255

    attack_steps = [1, 5, 10, 20, 50]
    epsilons = [1 / 255, 2 / 255, 4 / 255, 6 / 255, 8 / 255]
    # for attack_step in attack_steps:
    # for eps in epsilons:

    ### Your code here for creating the attacker object
    # Note that FGSM attack is a special case of PGD attack with specific hyper-parameters
    # You can also implement a separate FGSM class if you want
    attacker = attack_util.PGDAttack(
        attack_step=args.attack_step, eps=eps, alpha=alpha, loss_type=args.loss_type,
        targeted=args.targeted, num_classes=num_classes)
    ### Your code ends

    total = 0
    clean_correct_num = 0
    robust_correct_num = 0
    target_label = 1  ## only for targeted attack

    ## Make sure the model is in `eval` mode.
    model.eval()
    print("***********************************")
    print("total batches : ", len(test_loader.dataset))
    print("loss func : ", args.loss_type)
    print("targeted : ", args.targeted)
    print("epsilon : ", args.eps)
    print("alpha : ", args.alpha)
    print("attack_step : ", args.attack_step)
    print("***********************************")
    sample_no = 0

    for data, labels in tqdm(test_loader):
        # data: => [n, 3, 32, 32]
        # labels => [n]
        sample_no += 1
        # print("sample # : ", sample_no)

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
    # plot_accuracy([0.5569, 0.0072, 0.0001, 0.0, 0.0], [1, 5, 10, 20, 50], [0.762, 0.5433, 0.5273, 0.5216, 0.5205], 'T')
    # plot_accuracy([0.6016, 0.1172, 0.0068, 0.0008, 0.0001], ['1/255', '2/255', '4/255', '6/255', '8/255'],
    #               [0.7914, 0.7609, 0.6911, 0.6116, 0.5273], 'ϵ', colors=['red', 'blue'])
