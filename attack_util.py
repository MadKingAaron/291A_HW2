import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

### Do not modif the following codes
class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False
        
def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}

def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False

def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]
### Ends


### PGD Attack
class PGDAttack():
    def __init__(self, attack_step=10, eps=8 / 255, alpha=2 / 255, loss_type='ce', targeted=True, 
                 num_classes=10):
        '''
        attack_step: number of PGD iterations
        eps: attack budget
        alpha: PGD attack step size
        '''
        ### Your code here
        self.cross_entropy = nn.CrossEntropyLoss()
        
        self.targeted = targeted

        self.attack_step = attack_step
        self.epsilon = eps
        self.alpha = alpha
        
        self.loss_type = loss_type

        self.tau = 1.0

        self.targeted_class = 1

        
        ### Your code ends

    def l_inf_project(self, delta):
        return torch.clamp(delta, -self.epsilon, self.epsilon)

    def ce_loss(self, logits, y):
        ### Your code here
        #logits_unsq = torch.unsqueeze(logits, dim=0)
        mult_factor = -1
        
        if self.targeted:
            old_y = y
            y = torch.ones_like(old_y) * self.targeted_class
            mult_factor = 1
    
        loss = self.cross_entropy(logits, y)
        loss *= mult_factor

        return loss
        ### Your code ends

    def cw_loss(self, logits, y):
        ### Your code here
        y_dev = 'cpu' if y.get_device() < 0 else y.get_device()
        if self.targeted:
            old_y = y
            y = torch.ones_like(old_y) * self.targeted_class
            y = y.to(y_dev)
        
        #print("Logits Device:",logits.get_device())
        one_hot = torch.eye(len(logits[0])).to(y_dev)
        one_hot_labels = one_hot[y]
        largest, _ = torch.max((1-one_hot_labels)*logits, dim=1) # Largest
        second = torch.masked_select(logits, one_hot_labels.bool()) # get the second largest 

        if self.targeted:
            l = torch.clamp((largest-second), min=-0.0)
        else:
            l = torch.clamp((second-largest), min=-0.0)


        return l.sum()
        ### Your code ends
    

    def loss_func(self):
        return self.ce_loss if self.loss_type == 'ce' else self.cw_loss
    
   

    def perturb(self, model: nn.Module, X, y):
        delta = torch.zeros_like(X, requires_grad=True)

        
        ### Your code here
        X = X.clone().detach()
        y = y.clone().detach()

        X.requires_grad = True
        lf = self.loss_func()
        #print(lf.__name__)
        #y.requires_grad = True
        #print("Total Attack Steps: ", self.attack_step)
        
        for step in range(self.attack_step):
            delta = delta.clone().detach()
            delta.requires_grad = True

            # Get logits for perturbed input
            logits = model(X + delta)
            

            # Get loss
            loss = lf(logits, y)

            # print("Loss:", loss.item())
            
            # Backprop
            loss.backward()

            # Calc new delta
            delta = delta - (self.alpha * torch.sign(delta.grad))


            
            # Projection
            delta = self.l_inf_project(delta)

            # Zero Grad IGNORE!!!!!!!!
            #print(delta.grad is None)
            #delta.grad.data.zero_() 




        
        ### Your code ends
        
        return delta


### FGSMAttack
'''
Technically you can transform your PGDAttack to FGSM Attack by controling parameters like `attack_step`. 
If you do that, you do not need to implement FGSM in this class.
'''
class FGSMAttack():
    def __init__(self, eps=8 / 255, loss_type='ce', targeted=True, num_classes=10):
        self.eps = eps
        self.loss_type = loss_type
        self.targeted = targeted
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def l_inf_project(self, delta):
        return torch.clamp(delta, -self.epsilon, self.epsilon)

    def ce_loss(self, logits, y):
        ### Your code here
        #logits_unsq = torch.unsqueeze(logits, dim=0)
    
        loss = self.cross_entropy(logits, y)
        loss *= -1

        return loss
        ### Your code ends

    def cw_loss(self, logits, y):
        ### Your code here
        one_hot_labels = torch.eye(len(logits[0]))[y]
        largest, _ = torch.max((1-one_hot_labels)*logits, dim=1) # Largest
        second = torch.masked_select(logits, one_hot_labels.bool()) # get the second largest 

        if self.targeted:
            l = torch.clamp((largest-second), min=-0.0)
        else:
            l = torch.clamp((second-largest), min=-0.0)


        return l.sum()
    
    def get_loss_func(self):
        return self.ce_loss if self.loss_type == 'ce' else self.cw_loss

    def perturb(self, model: nn.Module, X, y):
        delta = torch.ones_like(X)
        ### Your code here
        loss_func = self.get_loss_func()

        X.requires_grad = True
        logits = model(X)

        loss = loss_func(logits, y)
        
        loss.backward()

        delta = (-1 * self.eps) * torch.sign(X.grad)

        ### Your code ends
        return delta
