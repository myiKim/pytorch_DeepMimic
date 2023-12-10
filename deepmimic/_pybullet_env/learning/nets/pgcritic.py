import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "pg_critic"

class PGCritic(nn.Module):
    def __init__(self, 
                 input_tensors, 
                 state_norm = None, #일단 goal은끄고 생각.
                 goal_norm = None,
                 val_norm = None,
                 len_input=None
                 ):
        
        super(PGCritic, self).__init__()
        self.layers = [1024, 512]        
        assert isinstance(input_tensors, list) == True
        if len_input is None:
            self.inputdim = len(input_tensors) * input_tensors[0].shape[-1]
        else:
            self.inputdim = len_input * input_tensors[0].shape[-1]       
        
        # For normalization
        self.state_norm = state_norm
        self.goal_norm = goal_norm
        self.val_norm = val_norm
        self.out_size = (0,)
        self.has_goal = False

        # Define the layers of the neural network
        
        self.fc1 = nn.Linear(self.inputdim, self.layers[0])
        self.fc2 = nn.Linear(self.layers[0], self.layers[1])
        self.out_layer = nn.Linear(self.layers[1], 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out_layer.weight) 
        nn.init.constant_(self.fc1.bias, 0) 
        nn.init.constant_(self.fc2.bias, 0) 
        nn.init.constant_(self.out_layer.bias, 0)  # Initialize bias to zero

    def forward(self, state_pt, goal_pt=None): #s_pt ==> state_pt

        norm_s_pt = self.state_norm.normalize_pt(state_pt) #s_norm이 바뀌어도 되는지 체크함 (nn상에서 변화가 같이 생기는지)
        
        input_tensors = [norm_s_pt]
        if self.has_goal and self.goal_norm and goal_pt:
            norm_g_pt = self.goal_norm.normalize_pt(goal_pt)
            input_tensors += [norm_g_pt]
        # Concatenate input tensors
        input_pt = torch.cat(input_tensors, dim=-1)        
        
        # Pass through the layers
        h = F.relu(self.fc1(input_pt))
        h = F.relu(self.fc2(h))
        norm_val_pt = self.out_layer(h)
        # print("norm_val_pt : ", norm_val_pt)
        val_pt = self.val_norm.unnormalize_pt(norm_val_pt.view(-1))
        # print("norm_a_pt : ", val_pt)

        return val_pt

def build_net(net_name, 
              input_tensor, 
              state_norm = None, #일단 goal은끄고 생각.
              goal_norm = None,
              val_norm = None,
              reuse=False):

  net = PGCritic(input_tensor, state_norm, goal_norm, val_norm)
#   print("PGActor :", net_name, " ==", net)
  return net 
