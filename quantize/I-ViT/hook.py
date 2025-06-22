import torch
import torch.nn as nn
import torch.nn.quantized as nnq


def get_activations(
    model: nn.Module, input: torch.Tensor, layers=None
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if layers is None:
        layers = (
            nn.Conv2d,
            nn.Linear,
            nn.LayerNorm,
            nn.Softmax,
            nn.GELU,
        )

    inputs = {}
    outputs = {}

    def _get_act(name):
        def hook(module, input, output):
            inputs[name] = input[0].detach()
            outputs[name] = output.detach()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, layers):
            module.register_forward_hook(_get_act(name))
    model(input)
    return inputs, outputs


def get_weights(model, layers=None, quantized_layers=None) -> dict[str, torch.Tensor]:
    if layers is None:
        layers = (nn.Conv2d, nn.Linear)
    if quantized_layers is None:
        quantized_layers = (nnq.Conv2d, nnq.Linear)

    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, layers):
            weights[name] = module.weight.data
        elif isinstance(module, quantized_layers):
            weights[name] = module.weight()
    return weights
