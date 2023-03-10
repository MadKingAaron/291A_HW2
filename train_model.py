import torch
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, Module
import attack_util
from attack_util import ctx_noparamgrad
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


class ModelTrainer():
    def __init__(self, model:Module, adv_gen, trainloader, lr, optimizer, device = 'cpu') -> None:
        self.adv_gen = adv_gen
        
        self.trainloader = trainloader

        self.device = device

        self.set_model(model, optimizer, lr)
        self.train_loss = CrossEntropyLoss()

        
    

    def set_model(self, model, optimizer:str, lr:float):
        self.model = model.to(self.device)

        if optimizer.lower() == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=lr)
        elif optimizer.lower() == 'adam':
            self.optimizer = Adam(model.parameters(), lr=lr)
    
    def save_model(self, save_dir:str):
        torch.save(self.model.state_dict(), save_dir)

    def train(self, epochs, valloader = None, log_dir:str=None):
        tb_writer = None
        if log_dir:
            tb_writer = SummaryWriter(log_dir=log_dir)
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for data, i in enumerate(self.trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                delta = self.adv_gen.perturb(self.model, inputs, labels)
                inputs_hat = inputs + delta

                self.optimizer.zero_grad()

                preds = self.model(inputs_hat)

                loss = self.loss_func(preds, labels)
                epoch_loss += loss.item()

                loss.backwards()
                self.optimizer.step()
            
            # Validate model, if valloader provided
            if valloader:
                clean_accuracy, robust_accuracy = self.test_model_accuracy(valloader)
                if tb_writer:
                    tb_writer.add_scalar("Loss/train", epoch_loss, loss)
                    tb_writer.add_scalar("CleanAccuracy/train", clean_accuracy, epoch)
                    tb_writer.add_scalar("RobustAccuracy/train", robust_accuracy, epoch)
        
        if tb_writer:
            tb_writer.flush()
            tb_writer.close()

    def test_model_accuracy(self, testloader):
        total_size = 0
        clean_correct_num = 0
        robust_correct_num = 0
        for data, i in enumerate(testloader):
            inputs, labels = data
    
            total_size += inputs.size(0)


            with ctx_noparamgrad(self.model):
                ### clean accuracy
                predictions = self.model(inputs)
                clean_correct_num += torch.sum(torch.argmax(predictions, dim = -1) == labels).item()
                
                ### robust accuracy
                # generate perturbation
                perturbed_data = self.adv_gen.perturb(self.model, data, labels) + inputs
                # predict
                predictions = self.model(perturbed_data)
                robust_correct_num += torch.sum(torch.argmax(predictions, dim = -1) == labels).item()
        
        return clean_correct_num/total_size, robust_correct_num/total_size
    