import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, Module
import attack_util
from attack_util import ctx_noparamgrad
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from data_util import augment_images


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
        
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer)
    
    def save_model(self, save_dir:str):
        self.model.save(save_dir)

    def train(self, epochs, valloader = None, log_dir:str=None, save_freq:int = 20):
        tb_writer = None
        if log_dir:
            tb_writer = SummaryWriter(log_dir=log_dir)
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for i, batch in enumerate(self.trainloader):
                inputs, labels = batch
                
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                delta = self.adv_gen.perturb(self.model, inputs, labels)
                inputs_hat = inputs + delta

                self.optimizer.zero_grad()

                preds = self.model(inputs_hat)

                loss = self.train_loss(preds, labels)
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()
            
            # Validate model, if valloader provided
            
            if valloader:
                clean_accuracy, robust_accuracy, val_robust_loss = self.test_model_accuracy(valloader)
                self.scheduler.step(val_robust_loss)
                if tb_writer:
                    tb_writer.add_scalar("Loss/train", epoch_loss, epoch)
                    tb_writer.add_scalar("Loss/val_robust", val_robust_loss, epoch)
                    tb_writer.add_scalar("CleanAccuracy/train", clean_accuracy, epoch)
                    tb_writer.add_scalar("RobustAccuracy/train", robust_accuracy, epoch)
            
            # Save checkpoint model after save_freq
            if (epoch+1) % save_freq == 0:
                self.save_model('./checkpoints/checkpoint_{}.pth'.format(epoch+1))
                
        if tb_writer:
            tb_writer.flush()
            tb_writer.close()

    def test_model_accuracy(self, testloader):
        total_size = 0
        clean_correct_num = 0
        robust_correct_num = 0
        total_robust_val_loss = 0
        for i, batch in enumerate(testloader):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
    
            total_size += inputs.size(0)


            with ctx_noparamgrad(self.model):
                ### clean accuracy
                predictions = self.model(inputs)
                clean_correct_num += torch.sum(torch.argmax(predictions, dim = -1) == labels).item()
                
                ### robust accuracy
                # generate perturbation
                perturbed_data = self.adv_gen.perturb(self.model, inputs, labels) + inputs
                # predict
                predictions = self.model(perturbed_data)
                robust_correct_num += torch.sum(torch.argmax(predictions, dim = -1) == labels).item()

                # Robust loss
                robust_loss = self.train_loss(predictions, labels).item()
                total_robust_val_loss += robust_loss

        
        return clean_correct_num/total_size, robust_correct_num/total_size, robust_loss


class TRADESModelTrainer(ModelTrainer):
    def __init__(self, model: Module, adv_gen, trainloader, lr, optimizer, device='cpu') -> None:
        super().__init__(model, adv_gen, trainloader, lr, optimizer, device)

    def train(self, epochs, gamma, valloader = None, log_dir:str=None, save_freq:int = 20, semi_supervised:bool = False):
        tb_writer = None
        if log_dir:
            tb_writer = SummaryWriter(log_dir=log_dir)
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for i, batch in enumerate(self.trainloader):
                inputs, labels = batch
                if semi_supervised:
                    aug_inputs = torch.cat((inputs, augment_images(inputs))).to(self.device)
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                preds = self.model(inputs)
                if semi_supervised:
                    pert_preds = self.model(aug_inputs)
                    pert_inputs = aug_inputs

                else:
                    pert_preds = preds
                    pert_inputs = inputs

                delta = self.adv_gen.perturb(self.model, pert_inputs, pert_preds)
                
                inputs_hat = pert_inputs + delta

                self.optimizer.zero_grad()

                preds_hat = self.model(inputs_hat)

                loss_hat = self.train_loss(preds_hat, pert_preds)
                loss_preds = self.train_loss(preds, labels)

                loss = (gamma * loss_preds) + loss_hat
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()
            
            # Validate model, if valloader provided
            if valloader:
                clean_accuracy, robust_accuracy = self.test_model_accuracy(valloader)
                if tb_writer:
                    tb_writer.add_scalar("Loss/train", epoch_loss, loss)
                    tb_writer.add_scalar("CleanAccuracy/train", clean_accuracy, epoch)
                    tb_writer.add_scalar("RobustAccuracy/train", robust_accuracy, epoch)
            
            # Save checkpoint model after save_freq
            if (epoch+1) % save_freq == 0:
                self.save_model('./checkpoints/checkpoint_{}.pth'.format(epoch+1))
        
        if tb_writer:
            tb_writer.flush()
            tb_writer.close()