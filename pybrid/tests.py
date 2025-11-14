"""A module with different tests for DoubleAmortModel models"""

import torch
import numpy as np
from pybrid.models import DoubleAmortModel


def test_model(
    img_batch: torch.Tensor,
    context_batch: torch.Tensor,
    model: DoubleAmortModel,
    layer: int = 0,
    num_iters: int = 100,
    init_std: float = 0.05,
    fixed_preds: bool = False,
    use_amort: bool = False,
    use_infer: bool = True,
) -> np.ndarray:
    """A function to test a model on a batch of images.

    Args:
        img_batch (torch.Tensor): A batch of images.
        model (HybridModel): A model to test.
        layer (int): Layer to test.
        num_iters (int): Number of iterations to test.
        init_std (float): Initial standard deviation for the model.
        fixed_preds (bool): Whether to use fixed predictions.
        use_amort (bool): Whether to use amortised inference.
        use_infer (bool): Whether to use inference at all (useful to test amortization only).
        thresh (Optional[float]): Threshold for amortised inference.
        delta_thresh (Optional[float]): Delta threshold for amortised inference.

    Returns:
        A torch.Tensor with of dimensions (batch_size, num_iters, layer_length)."""

    # reset model
    model.reset()
    if use_amort:
        model.set_img_batch_amort(img_batch)
        model.forward_mu(context_batch)
    else:
        model.reset_mu(img_batch.size(0), init_std)
    model.set_img_batch(img_batch)

    # preallocate tensor
    layer_length = model.nodes[layer]
    layer_activations = torch.zeros(
        (img_batch.size(0), num_iters, layer_length), device=img_batch.device
    )

    for n in range(1, model.num_nodes):
        model.preds[n] = model.layers[n - 1].forward(model.mus[n - 1])
        model.errs[n] = model.mus[n] - model.preds[n]

    for itr in range(num_iters):
        if use_infer:
            delta = model.layers[0].backward(model.errs[1])
            model.mus[0] = model.mus[0] + model.mu_dt * (2 * delta)
            for l in range(1, model.num_layers):
                delta = model.layers[l].backward(model.errs[l + 1]) - model.errs[l]
                model.deltas[l] = delta
                model.mus[l] = model.mus[l] + model.mu_dt * (2 * delta)

            for n in range(1, model.num_nodes):
                if not fixed_preds:
                    model.preds[n] = model.layers[n - 1].forward(model.mus[n - 1])
                model.errs[n] = model.mus[n] - model.preds[n]

        # save layer activations
        layer_activations[:, itr] = model.mus[layer]

    return layer_activations.cpu().numpy()
