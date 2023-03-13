import argparse

import train_model

import data_util
import model_util
import attack_util


def get_model(norm_layer):
    model = model_util.ResNet18(num_classes=10)
    model.normalizer = norm_layer
    return model


def parse_args():
    '''Parse input arguments'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=50, help='Number of training epochs'
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help='Learning rate for training'
    )
    parser.add_argument(
        "--optimizer", type=str, default='sgd', choices=['sgd', 'adam'], help="Optimizer for training"
    )
    parser.add_argument(
        "--eps", type=int, default=8, help="Attack budget: epsilon / 255"
    )
    parser.add_argument(
        "--beta", type=float, default=2, help="PGD attack step size: beta / 255"
    )
    parser.add_argument(
        "--attack_step", type=int, default=10, help="Number of PGD iterations"
    )
    parser.add_argument(
        "--attack_loss_type", type=str, default="ce", choices=['ce','cw'], help="Loss type for attack"
    )
    parser.add_argument(
        '--data_dir', default='./data/', type=str, help="Folder to store downloaded dataset"
    )
    parser.add_argument("--checkpt_freq", type=int, default=20, help="Frequency to save model checkpoints")

    parser.add_argument("--trades", action='store_true')

    parser.add_argument("--unsupervised", action='store_true')

    parser.add_argument("--gamma", type=float, default=0.001, help="Gamma value for TRADES")

    parser.add_argument(
        '--save_model', action='store_true'
    )
    parser.add_argument(
        '--save_path', default='resnet_cifar10.pth', help='Filepath to save trained model'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./run/training_log', help="Output directory for TensorBoard"
    )

    parser.add_argument("--load_checkpoint", type=str, default="", help="(Optional) Load checkpoint dir")
    parser.add_argument("--targeted", action='store_true')
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Get model and dataset
    train_loader, val_loader, test_loader, norm_layer = data_util.cifar10_dataloader(data_dir=args.data_dir)
    model = get_model(norm_layer)

    if args.load_checkpoint != "":
        model = model.load(args.load_checkpoint, args.device)
    
    model = model.to(args.device)

    # Get attacker
    eps = args.eps / 255
    beta = args.beta / 255
    if args.attack_step > 1:
        attacker = attack_util.PGDAttack(
            attack_step=args.attack_step, eps=eps, alpha=beta, loss_type=args.attack_loss_type,
            targeted=args.targeted, num_classes=10)
    else: # Get FGSM Attack
        attacker = attack_util.FGSMAttack(eps=eps, loss_type=args.attack_loss_type, targeted=args.targeted,
            num_classes=10)
    


    if args.trades:
        trainer = train_model.TRADESModelTrainer(model, attacker, train_loader, args.lr, args.optimizer, device=args.device)
        # Train model
        trainer.train(epochs=args.epochs, valloader=val_loader, log_dir=args.log_dir, gamma=args.gamma, semi_supervised=args.unsupervised)
    else:
        # Get trainer object
        trainer = train_model.ModelTrainer(model, attacker, train_loader, args.lr, args.optimizer, device=args.device)
        # Train model
        trainer.train(epochs=args.epochs, valloader=val_loader, log_dir=args.log_dir)
        

    trainer.save_model(args.save_path)

if __name__ == "__main__":
    main()


