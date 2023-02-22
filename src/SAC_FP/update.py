"""
SAC network update functions
"""

# import libraries
import equinox as eqx
import optax


@eqx.filter_jit
def update_model(params, grads):
    model, optimizer, opt_state = params
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, optimizer, opt_state


def update_vf_target(base, target, **kwargs):
    tau = kwargs.get('tau', 4e-3)

    for idx, (base_layer, target_layer) in enumerate(zip(base.general_layers, target.general_layers)):
        weight = base_layer.weight * tau + target_layer.weight * (1 - tau)
        bias = base_layer.bias * tau + target_layer.bias * (1-tau)

        target = eqx.tree_at(lambda model: model.general_layers[idx].weight, target, replace=weight)
        target = eqx.tree_at(lambda model: model.general_layers[idx].bias, target, replace=bias)
    return target