import torch
import torch.nn as nn
from typing import Dict

class HelloModel(nn.Module):
    def __init__(self, hello_str="hello world"):
        super().__init__()
        self.hello_str = hello_str
        print(f"hello_str is {hello_str}")
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]):
        print(f"predict_action, {self.hello_str}")
        print(obs_dict['left_wrist_img'].shape, type(obs_dict['left_wrist_img']))
        device = obs_dict['left_wrist_img'].device
        
        return torch.zeros([2,10], device=device)
