import torch
import torch.nn as nn

"""
    Input:
        - net: model to be pruned
        - u: coefficient that determines the pruning threshold
    Output:
        None (in-place modification on the model)
"""

def CLP(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                if idx >= conv.weight.shape[0]:
                    continue
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = params[name+'.weight'].mean()
            params[name+'.bias'][index] = params[name+'.bias'].mean()
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)


def CLP_head(net, u):
    params = net.state_dict()
    num_heads = net.blocks[0].attn.num_heads
    head_dim = net.blocks[0].attn.head_dim
    
    for name, m in net.named_modules():
        head_lips = []
        if 'qkv' in name:
            head_weights = m.weight.reshape(3, num_heads, head_dim, -1)

            for i in range(3):
                for j in range(num_heads):
                    weight = head_weights[i][j]
                    head_lips.append(torch.svd(weight)[1].max())
            head_lips = torch.Tensor(head_lips)
            index = torch.where(head_lips>head_lips.mean() + u*head_lips.std())[0]

            for i in index:
                qkv_index = i//num_heads
                heads_index = i%num_heads
                head_weights[qkv_index][heads_index].data *= 0.5