import torch

from solver import find_c_adam, find_c_sgd


# new group for each param (so dynamic setting is easier)
def abc_parametrization(model, al, bl, cl, lr_prefactor, std_prefactor):
    from model import Embedding, LayerNorm, MLPLayer

    lr_scale_groups = []

    embed_a, embed_b, embed_c = al[0], bl[0], cl[0]
    hidden_a, hidden_b, hidden_c = al[1], bl[1], cl[1]
    readout_a, readout_b, readout_c = al[2], bl[2], cl[2]

    def setup_parametrization(layer, a, b, c):
        if isinstance(layer, LayerNorm):
            weight = layer.ln.weight
            n = 1 # TODO: check if this is correct?
        elif isinstance(layer, Embedding):
            weight = layer.params
            n = weight.size(1)
        elif isinstance(layer, MLPLayer):
            weight = layer.lin.weight
            n = weight.size(1)
        else:
            raise ValueError()

        l_mult = n ** -a
        var_l = n ** (-2*b)
        std=std_prefactor * (var_l ** 0.5)
        lr_scale = n ** -c

        if isinstance(layer, Embedding) and b == 0:  # special case from Everett et al.
            std = 0.01

        if isinstance(layer, MLPLayer):  # TODO: add embedding and layer norm?
            lr_scale_groups.append((lr_scale, weight))

        layer.layer_multiplier = l_mult
        torch.nn.init.normal_(weight, mean=0.0, std=std)

    def traverse_model(module):
        for name, layer in module.named_children():
            if 'embed' in name or isinstance(layer, Embedding) or isinstance(layer, LayerNorm):
                setup_parametrization(layer, a=embed_a, b=embed_b, c=embed_c)
            elif 'readout' in name:
                setup_parametrization(layer, a=readout_a, b=readout_b, c=readout_c)
            elif isinstance(layer, MLPLayer):
                setup_parametrization(layer, a=hidden_a, b=hidden_b, c=hidden_c)
            else:
                traverse_model(layer)

    traverse_model(model)
    optim_groups = [{'params': params, 'lr': lr_prefactor * lr_scale} for lr_scale, params in lr_scale_groups]
    return optim_groups


def maximal_lr_scheduler(optimizer, n, al, bl, lr_prefactor=0.1, feature_learning=False):
    def _compute_cl(alpha_l, omega_l, u_l):
        # Find maximal lr exponents
        if isinstance(optimizer, torch.optim.AdamW):
            solver = find_c_adam
        elif isinstance(optimizer, torch.optim.SGD):
            solver = find_c_sgd
        else:
            raise ValueError(f"Unsupported optimizer: {type(optimizer)}")
        cl, rl = solver(a=al, b=bl, alpha=alpha_l, u=u_l, omega=omega_l, fl=feature_learning)
        return cl

    def _lr_adjuster(alpha_l, u_l, omega_l):
        # Compute c_l based on measured alignment
        cl = _compute_cl(alpha_l=alpha_l, omega_l=omega_l, u_l=u_l)
        # Dynamically adjust learning rates for each parameter group 
        for i, (param_group, c) in enumerate(zip(optimizer.param_groups, cl)):
            lr_scale = n ** -c
            param_group['lr'] = lr_prefactor * lr_scale
        return [param_group['lr'] for param_group in optimizer.param_groups]
    return _lr_adjuster    


def constant_lr_scheduler(optimizer):
    # just for logging
    def _lr_adjuster(alpha_l, u_l, omega_l):
        # just grab lrs from param groups
        return [param_group['lr'] for param_group in optimizer.param_groups]
    return _lr_adjuster