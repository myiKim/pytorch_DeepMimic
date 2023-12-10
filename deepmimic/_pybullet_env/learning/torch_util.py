import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# try:
#   xavier_initializer = tf.contrib.layers.xavier_initializer()
# except Exception:
#   xavier_initializer = None

xavier_initializer = nn.init.xavier_uniform_
PI = np.pi

def disable_gpu():
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
  return


def var_shape(x):
  # out = [k.value for k in x.get_shape()]
  # assert all(isinstance(a, int) for a in out), "shape function assumes that shape is fully known"
  out = list(x.size())
  return out


def intprod(x):
  return int(np.prod(x))


def numel(x):
  # n = intprod(var_shape(x))
  # return n
  return x.numel()


def flat_grad(loss, var_list, grad_ys=None):
  grads = torch.autograd.grad(loss, var_list, grad_outputs=grad_ys, create_graph=True)
  return torch.cat([grad.contiguous().view(-1) for grad in grads])
#   grads = tf.gradients(loss, var_list, grad_ys)
#   return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], axis=0)


def fc_net_old(input, layers_sizes, activation, reuse=None,
           flatten=False):  # build fully connected network
  curr_tf = input
#   for i, size in enumerate(layers_sizes):
#     with tf.variable_scope(str(i), reuse=reuse):
#       curr_tf = tf.layers.dense(inputs=curr_tf,
#                                 units=size,
#                                 kernel_initializer=xavier_initializer,
#                                 activation=activation if i < len(layers_sizes) - 1 else None)
#   if flatten:
#     assert layers_sizes[-1] == 1
#     curr_tf = tf.reshape(curr_tf, [-1])

  return curr_tf

def fc_net(input, layers_sizes= [1024, 512], activation =F.relu, flatten=False):
    curr_pt = input
    for i, size in enumerate(layers_sizes):
        # Define a linear layer
        linear_layer = nn.Linear(curr_pt.size(-1), size)
        
        # Apply the specified activation function
        if i < len(layers_sizes) - 1:
            curr_pt = activation(linear_layer(curr_pt))
        else:
            curr_pt = linear_layer(curr_pt)

    if flatten:
        assert layers_sizes[-1] == 1
        curr_tf = curr_tf.view(-1)

    return curr_pt

# def copy(sess, src, dst):
# #   assert len(src) == len(dst)
# #   sess.run(list(map(lambda v: v[1].assign(v[0]), zip(src, dst))))
#   return
def copy(src, dst, requires_grad=False):
  assert len(src) == len(dst)
  for src_param, dst_param in zip(src, dst):
    # dst_param.data.copy_(src_param.data)
    dst_param.data = torch.new_tensor(src_param)
    if not requires_grad: 
      dst_param.requires_grad = False


def flat_grad(loss, var_list):
  pass
#   grads = tf.gradients(loss, var_list)
#   return tf.concat(axis=0,
#                    values=[tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)])


# def calc_logp_gaussian(x_pt, mean_pt, std_pt):
#   dim = tf.to_float(tf.shape(x_tf)[-1])

#   if mean_tf is None:
#     diff_tf = x_tf
#   else:
#     diff_tf = x_tf - mean_tf

#   logp_tf = -0.5 * tf.reduce_sum(tf.square(diff_tf / std_tf), axis=-1)
#   logp_tf += -0.5 * dim * np.log(2 * np.pi) - tf.reduce_sum(tf.log(std_tf), axis=-1)

#   return logp_tf
def calc_logp_gaussian(x_pt, mean_pt, std_pt):
    # print("////////////////////"*21, "x_pt : ", x_pt.size())
    dimension = x_pt.size(-1)

    if mean_pt is None:
        diff_pt = x_pt
    else:
        diff_pt = x_pt - mean_pt

    logp_pt = -0.5 * torch.sum((diff_pt / std_pt) ** 2, dim=-1)
    logp_pt += -0.5 * dimension * torch.log(2 * torch.tensor(PI)) - torch.sum(torch.log(std_pt), dim=-1)

    return logp_pt



def calc_bound_loss(x_pt, bound_min, bound_max):
    # penalty for violating bounds
    compared_to = torch.zeros_like(x_pt)
    violation_min = torch.minimum(x_pt - bound_min, compared_to)
    violation_max = torch.maximum(x_pt - bound_max, compared_to)
    violation = torch.sum(torch.square(violation_min), axis=-1) + torch.sum(torch.square(violation_max), axis=-1)
    loss = 0.5 * torch.mean(violation, axis=-1)
    return loss


# def calc_bound_loss2(x_pt, bound_min, bound_max):
#     # Penalty for violating bounds
#     violation_min = torch.min(x_pt - bound_min, torch.zeros_like(x_pt))
#     print("1 : ", torch.sum(violation_min ** 2, dim=-1))
#     violation_max = torch.max(x_pt - bound_max, torch.zeros_like(x_pt))
#     print("2 : ",torch.sum(violation_max ** 2, dim=-1))
#     violation = torch.sum(violation_min ** 2, dim=-1) + torch.sum(violation_max ** 2, dim=-1)
#     loss = 0.5 * torch.mean(violation)
#     return loss


# def calc_bound_loss(x_tf, bound_min, bound_max):  
#   violation_min = tf.minimum(x_tf - bound_min, 0) #Returns the min of x and y (i.e. x < y ? x : y) element-wise.
#   violation_max = tf.maximum(x_tf - bound_max, 0)
#   violation = tf.reduce_sum(tf.square(violation_min), axis=-1) + tf.reduce_sum(
#       tf.square(violation_max), axis=-1)
#   loss = 0.5 * tf.reduce_mean(violation)
  # loss =0
  # return loss


class SetFromFlat(object):

  def __init__(self, sess, var_list, dtype=torch.float32):
    # assigns = []
    # shapes = list(map(var_shape, var_list))
    # total_size = np.sum([intprod(shape) for shape in shapes])

    # self.sess = sess
    # self.theta = tf.placeholder(dtype, [total_size])
    # start = 0
    # assigns = []

    # for (shape, v) in zip(shapes, var_list):
    #   size = intprod(shape)
    #   assigns.append(tf.assign(v, tf.reshape(self.theta[start:start + size], shape)))
    #   start += size

    # self.op = tf.group(*assigns)

    return

  def __call__(self, theta):
    # self.sess.run(self.op, feed_dict={self.theta: theta})
    return


class GetFlat(object):

  def __init__(self, sess, var_list):
    # self.sess = sess
    # self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
    return

  def __call__(self):
    # return self.sess.run(self.op)
    return



# import torch
# import torch.nn as nn
# import numpy as np
# import os

# xavier_initializer = nn.init.xavier_uniform_

# def disable_gpu():
#     os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# def var_shape(x):
#     return list(x.size())

# def intprod(x):
#     return int(np.prod(x))

# def numel(x):
#     return x.numel()

# def flat_grad(loss, var_list, grad_ys=None):
#     grads = torch.autograd.grad(loss, var_list, grad_outputs=grad_ys, create_graph=True)
#     return torch.cat([grad.contiguous().view(-1) for grad in grads])

# def fc_net(input, layers_sizes, activation, flatten=False):
#     curr_pt = input
#     for i, size in enumerate(layers_sizes):
#         layer = nn.Linear(curr_pt.size(-1), size)
#         nn.init.xavier_uniform_(layer.weight)
#         curr_pt = layer(curr_pt)
#         if i < len(layers_sizes) - 1:
#             curr_pt = activation(curr_pt)
#     if flatten:
#         assert layers_sizes[-1] == 1
#         curr_pt = curr_pt.view(-1)
#     return curr_pt

# def copy(src, dst):
#     assert len(src) == len(dst)
#     for src_param, dst_param in zip(src, dst):
#         dst_param.data.copy_(src_param.data)

# def calc_logp_gaussian(x_pt, mean_pt, std_pt):
#     dim = float(x_pt.size(-1))

#     if mean_pt is None:
#         diff_pt = x_pt
#     else:
#         diff_pt = x_pt - mean_pt

#     logp_pt = -0.5 * torch.sum((diff_pt / std_pt) ** 2, dim=-1)
#     logp_pt += -0.5 * dim * np.log(2 * np.pi) - torch.sum(torch.log(std_pt), dim=-1)

#     return logp_pt

# def calc_bound_loss(x_pt, bound_min, bound_max):
#     violation_min = torch.clamp_min(x_pt - bound_min, min=0)
#     violation_max = torch.clamp_max(x_pt - bound_max, max=0)
#     violation = torch.sum(violation_min ** 2, dim=-1) + torch.sum(violation_max ** 2, dim=-1)
#     loss = 0.5 * torch.mean(violation)
#     return loss

# class SetFromFlat(object):
#     def __init__(self, var_list, dtype=torch.float32):
#         self.var_list = var_list

#     def __call__(self, theta):
#         start = 0
#         for v in self.var_list:
#             size = numel(v)
#             v.data.copy_(theta[start:start + size].view(v.size()))
#             start += size

# class GetFlat(object):
#     def __init__(self, var_list):
#         self.var_list = var_list

#     def __call__(self):
#         return torch.cat([v.contiguous().view(-1) for v in self.var_list])
