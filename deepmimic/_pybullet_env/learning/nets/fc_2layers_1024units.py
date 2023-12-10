# import _pybullet_env.deep_mimic.learning.tf_util as TFUtil
# import _pybullet_env.learning.torch_util as PTUtil

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "fc_2layers_1024units"

class FCNet2layersBasicUnits(nn.Module):
    def __init__(self, input_tensors, len_input=None): #test:ReinforcementLearning\testcodes\torch_net_builder_test.ipynb
        super(FCNet2layersBasicUnits, self).__init__()
        self.layers = [1024, 512]
        #인풋이 마치 input_tensors = [torch.ones(1, num_feats), torch.ones(1, num_feats), ..] 이래야함..
        assert isinstance(input_tensors, list) == True
        if len_input is None:
            self.inputdim = len(input_tensors) * input_tensors[0].shape[-1]
        else:
            self.inputdim = len_input * input_tensors[0].shape[-1]       

        # Define the layers of the neural network
        self.fc1 = nn.Linear(self.inputdim, self.layers[0])
        self.fc2 = nn.Linear(self.layers[0], self.layers[1])

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0) 
        nn.init.constant_(self.fc2.bias, 0) 

    def forward(self, input_tfs):
        # Concatenate input tensors
        input_tf = torch.cat(input_tfs, dim=-1)

        # Pass through the layers
        h = F.relu(self.fc1(input_tf))
        h = F.relu(self.fc2(h))

        return h

def build_net(input_tensor, reuse=False):
  # layers = [1024, 512]
  # activation = tf.nn.relu
  # input_tf = tf.concat(axis=-1, values=input_tfs)
  # h = TFUtil.fc_net(input_tf, layers, activation=activation, reuse=reuse)
  # h = activation(h)
  # return h  
#   out = FCNet2layersBasicUnits(input_tensor)     
  net = FCNet2layersBasicUnits(input_tensor)
#   print(out.out_features)
  return net # Return the created network instance
