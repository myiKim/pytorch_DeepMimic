import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "pg_actor"

class PGActor(nn.Module):
    def __init__(self, 
                 input_tensors, 
                 action_size, 
                 state_norm = None, #일단 goal은끄고 생각.
                 goal_norm = None,
                 action_norm = None,
                 init_output_scale=None,
                 len_input=None
                 ):
        
        super(PGActor, self).__init__()
        self.layers = [1024, 512]        
        assert isinstance(input_tensors, list) == True
        if len_input is None:
            self.inputdim = len(input_tensors) * input_tensors[0].shape[-1]
        else:
            self.inputdim = len_input * input_tensors[0].shape[-1]       
        
        # For normalization
        self.state_norm = state_norm
        self.goal_norm = goal_norm
        self.action_norm = action_norm
        # self.state_mean_pt = state_norm.mean_pt
        # self.state_std_pt = state_norm.std_pt
        self.out_size = (0,)
        self.has_goal = False

        # Define the layers of the neural network
        
        self.fc1 = nn.Linear(self.inputdim, self.layers[0])
        self.fc2 = nn.Linear(self.layers[0], self.layers[1])
        self.out_layer = nn.Linear(self.layers[1], action_size)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.out_layer.weight, -init_output_scale, init_output_scale)
        nn.init.constant_(self.fc1.bias, 0) 
        nn.init.constant_(self.fc2.bias, 0) 
        nn.init.constant_(self.out_layer.bias, 0)  # Initialize bias to zero

    def forward(self, state_pt, goal_pt=None): #s_pt ==> state_pt
        # print("state_pt : ", state_pt) #나중에 다 구현후 값체크 필요, 커멘트 켜서..
        norm_s_pt = self.state_norm.normalize_pt(state_pt) #s_norm이 바뀌어도 되는지 체크함 (nn상에서 변화가 같이 생기는지)
        # print("norm_s_pt : ", norm_s_pt)

        input_tensors = [norm_s_pt]
        if self.has_goal and self.goal_norm and goal_pt:
            norm_g_pt = self.goal_norm.normalize_pt(goal_pt)
            input_tensors += [norm_g_pt]
        # Concatenate input tensors
        input_pt = torch.cat(input_tensors, dim=-1)        
        
        # Pass through the layers
        h = F.relu(self.fc1(input_pt))
        h = F.relu(self.fc2(h))
        action_param_pt = self.out_layer(h)
        # print("action_param_pt (a_pt) : ", action_param_pt)
        norm_a_pt = self.action_norm.unnormalize_pt(action_param_pt)
        # print("norm_a_pt : ", norm_a_pt)

        return norm_a_pt

def build_net(net_name, 
              input_tensor, 
              action_size, 
              state_norm = None, #일단 goal은끄고 생각.
              goal_norm = None,
              action_norm = None,
              init_output_scale = 0.1, reuse=False):

  net = PGActor(input_tensor, action_size, state_norm, goal_norm, action_norm, init_output_scale = init_output_scale)
#   print("PGActor :", net_name, " ==", net)
  return net 
