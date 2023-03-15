import argparse
import os

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default='resnet_cifar10.pth', help='Filepath to the trained model'
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    print("Testing Model: {}".format(args.model_path))

    print("\n\nCE Loss:")
    os.system("python evaluate.py --device cuda --alpha 8 --eps 2 --attack_step 50 --loss_type ce --model_path {}".format(args.model_path))

    print("\n\nCW Loss:")
    os.system("python evaluate.py --device cuda --alpha 8 --eps 2 --attack_step 50 --loss_type cw --model_path {}".format(args.model_path))

if __name__ == "__main__":
    main()