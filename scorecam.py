import json
import argparse
from datetime import datetime
import calendar
import numpy as np
import torch
import torch.nn.functional as F



class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    
class ScoreCAM():
    """ Score CAM """

    def __init__(self, model, target_layer, n_batch=32):
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """
        self.model=model
        self.target_layer = target_layer
        self.values = SaveValues(self.target_layer)
        self.n_batch = n_batch
        self.device=model.device
        

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: the index of the target class
        Return:
            heatmap: class activation mappings of predicted classes
        """

        with torch.no_grad():
            _, _,  W = x.shape

            self.model.zero_grad()
            output = self.model(x)
            #print("score",output)
            #prob = F.softmax(score, dim=1)

            if idx is None:
                p, idx = torch.max(output, dim=1)
                idx = idx.item()
                # print("predicted class ids {}\t probability {}".format(idx, p))

            # # calculate the derivate of probabilities, not that of scores
            # prob[0, idx].backward(retain_graph=True)
            self.activations = self.values.activations.to('cpu').clone()
            # put activation maps through relu activation
            # because the values are not normalized with eq.(1) without relu.
            #print("self.activations",            self.activations)
            self.activations = F.relu(self.activations)
            #print(self.activations.shape)
            self.activations = F.interpolate(
                self.activations, W, mode='linear')
            _, C, _ = self.activations.shape
            #print("self.activations",self.activations.shape)
            # normalization
            act_min, _ = self.activations.view(1, C, -1).min(dim=2)
            act_min = act_min.view(1, C, 1)          
            act_max, _ = self.activations.view(1, C, -1).max(dim=2)
            act_max = act_max.view(1, C, 1)
            
            
            denominator = torch.where(
                (act_max - act_min) != 0., act_max - act_min, torch.tensor(1.)
            )

            self.activations = self.activations / denominator

            # generate masked images and calculate class probabilities
            probs = []
            #print("self.activations",self.activations.shape)
            mask = self.activations[:, ].transpose(0, 1)
            #print("mask.shape",mask.shape)
            mask = mask.to(self.device)
            masked_x = x * mask
            #print("masked_x",masked_x.shape)
            score = self.model(masked_x)
            #probs.append(F.softmax(score, dim=1)[:, idx].to('cpu').data)
            probs.append(score[:, idx].to('cpu').data)

            probs = torch.stack(probs)
            weights = probs.view(1, C, 1)
            #print("probs",probs)
            # shape = > (1, 1, H, W)
            cam = (weights * self.activations).sum(1, keepdim=True)
            cam = F.relu(cam)
            cam -= torch.min(cam)
            cam /= torch.max(cam)
            
            #print("cam",cam)
        return output,cam.data

    def __call__(self, x):
        return self.forward(x)

def scorecam(model,targetLayer,test_data):

    wrapped_model=ScoreCAM(model,targetLayer)
    out_mask=np.zeros((len(test_data),test_data.mz_range))
    score_cam_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=8)
        
    j=0
    for batch_idx, (data, target) in enumerate(score_cam_loader):
        data=data.float().to(model.device)
        output,mask = wrapped_model(data)
        out_mask[j,:]=mask.to('cpu')
        #out_seq[j,:]=data
        #print("mask[0]",mask.view(-1) )
        j=j+1
            # if idx == 0:
        #   break
    #print("cam image size:",cam_image.shape)
    
    return out_mask