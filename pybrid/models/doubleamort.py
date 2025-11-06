""" Class for a hybrid model with two amortization networks """

from typing import List
import torch
from pybrid import utils
from pybrid.models.base import BaseModel
from pybrid.layers import FCLayer


class DoubleAmortModel(BaseModel):
    """Hybrid model with two amortization networks"""

    def __init__(
        self,
        nodes: List[int],
        amort_nodes: List[int],
        act_fn: utils.Activation,
        n_amort_nets: int = 2,
        mu_dt: float = 0.01,
        use_bias: bool = False,
        kaiming_init: bool = False,
        device: str = "cpu",
    ):
        self.device = device
        self.nodes = nodes
        self.amort_nodes = amort_nodes
        self.n_amort_nets = n_amort_nets
        self.mu_dt = mu_dt

        self.num_nodes = len(nodes)
        self.num_layers = len(nodes) - 1
        self.total_params = 0

        self.layers = []
        # intialize empty lists for each amortization network
        self.amort_nets = [[] for _ in range(n_amort_nets)]
        for l in range(self.num_layers):
            layer_act_fn = utils.Linear() if (l == self.num_layers - 1) else act_fn
            layer = FCLayer(
                in_size=nodes[l],
                out_size=nodes[l + 1],
                act_fn=layer_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
            )
            self.layers.append(layer)
            self.total_params = self.total_params + (
                (nodes[l] * nodes[l + 1]) + nodes[l + 1]
            )

            # add the amortization layer to each amortization network
            for n in range(n_amort_nets):
                amort_layer = FCLayer(
                    in_size=amort_nodes[l],
                    out_size=amort_nodes[l + 1],
                    act_fn=layer_act_fn,  # TODO
                    use_bias=use_bias,
                    kaiming_init=kaiming_init,
                    is_amortised=True,
                )
                self.amort_nets[n].append(amort_layer)
        self.mean_weights, self.mean_biases = self.get_weight_stats()

    def reset(self):
        self.preds = [[] for _ in range(self.num_nodes)]
        self.errs = [[] for _ in range(self.num_nodes)]
        self.deltas = [[] for _ in range(self.num_nodes)]
        self.q_preds = [[] for _ in range(self.num_nodes)]
        self.q_errs = [[] for _ in range(self.num_nodes)]
        self.mus = [[] for _ in range(self.num_nodes)]

    def reset_mu(self, batch_size, init_std=0.05):
        for l in range(self.num_layers):
            tensor = torch.empty(batch_size, self.layers[l].in_size).normal_(
                mean=0, std=init_std
            )
            self.mus[l] = utils.set_tensor(tensor)

    def set_img_batch(self, img_batch):
        # clip image to desired size
        self.mus[-1] = img_batch.clone()[:, : self.nodes[-1]]

    def set_img_batch_amort(self, img_batch):
        self.q_preds[0] = img_batch.clone()

    def set_label_batch(self, label_batch):
        self.mus[0] = label_batch.clone()

    def forward(self, val, net_i):
        """Forward pass through an amortization network"""
        for l in range(1, self.num_nodes):
            # intialize holder_val with the expected size of the value after this layer
            holder_val = torch.zeros(
                net_i.shape[0], self.amort_nodes[l], device=self.device
            )
            # cycle through the different amortization networks in net_i
            for n in range(self.n_amort_nets):
                # determine the entries for this network in the batch
                batch_net = net_i == n
                holder_val[batch_net] = self.amort_nets[n][l - 1].forward(
                    val[batch_net]
                )
            val = holder_val  # copy for next layer

        return val

    def backward(self, val) -> torch.Tensor:
        """Backward pass through the inference network"""
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def forward_layer(self, max_layer, val, net_i):
        """Forward pass through the amortization network up to the max_layer-th layer"""
        for layer in self.amort_nets[net_i][:max_layer]:
            val = layer.forward(val)
        return val

    def backward_layer(self, starting_layer, val):
        """Backward pass through the inference network starting from the starting_layer-th layer"""
        for layer in self.layers[starting_layer:]:
            val = layer.forward(val)
        return val

    def forward_mu(self, net_i):
        """Forward pass through the amortization network to initialize in the inference network"""
        for l in range(1, self.num_nodes):
            # initialize q_preds with zeros
            self.q_preds[l] = torch.zeros(
                net_i.shape[0], self.amort_nodes[l], device=self.device
            )
            # cycle through the different amortization networks in net_i
            for n in range(self.n_amort_nets):
                # determine the entries for this network in the batch
                batch_net = net_i == n
                self.q_preds[l][batch_net] = self.amort_nets[n][l - 1].forward(
                    self.q_preds[l - 1][batch_net]
                )
            # self.q_preds[l] = self.amort_nets[net_i][l - 1].forward(self.q_preds[l - 1])
        mus = self.q_preds[::-1]
        for l in range(self.num_nodes):
            # trim the mus to the desired size
            self.mus[l] = mus[l].clone()[:, : self.nodes[l]]

    def backward_mu(self):
        for l in range(1, self.num_layers):
            self.mus[l] = self.layers[l - 1].forward(self.mus[l - 1])

    def train_batch(
        self,
        img_batch,
        label_batch,
        amort_net_i,
        num_iters=20,
        init_std=0.05,
        fixed_preds=False,
        use_amort=True,
        thresh=None,
        delta_thresh=None,
        no_backward=False,
        supervised=True,
        freeze_top=True,
    ):
        self.reset()
        if use_amort:
            self.set_img_batch_amort(img_batch)
            self.forward_mu(amort_net_i)
            self.set_img_batch(img_batch)
        else:
            if not no_backward:
                self.set_label_batch(label_batch)
                self.backward_mu()
                self.set_img_batch(img_batch)
            else:
                self.reset_mu(img_batch.size(0), init_std)
                self.set_img_batch(img_batch)

        if supervised:
            self.set_label_batch(label_batch)

        num_iter, avg_errs = self.train_updates(
            num_iters,
            fixed_preds=fixed_preds,
            thresh=thresh,
            delta_thresh=delta_thresh,
            freeze_top=freeze_top,
        )
        self.update_grads()
        if use_amort:
            self.update_amort_grads(amort_net_i)
        return num_iter, avg_errs

    def test_batch(
        self,
        img_batch,
        amort_net_i,
        num_iters=100,
        init_std=0.05,
        fixed_preds=False,
        use_amort=True,
        thresh=None,
        delta_thresh=None,
    ):
        self.reset()
        if use_amort:
            self.set_img_batch_amort(img_batch)
            self.forward_mu(amort_net_i)
        else:
            self.reset_mu(img_batch.size(0), init_std)

        self.set_img_batch(img_batch)
        num_iter, avg_errs = self.test_updates(
            num_iters, fixed_preds=fixed_preds, thresh=thresh, delta_thresh=delta_thresh
        )
        return self.mus[0], num_iter, avg_errs

    def train_updates(
        self,
        num_iters,
        fixed_preds=False,
        thresh=None,
        delta_thresh=None,
        freeze_top=True,
    ):
        for n in range(1, self.num_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        avg_errs = []
        avg_err = self.get_errors()[0] / self.total_params
        avg_errs.append(avg_err)

        avg_deltas = []
        itr = 0
        for itr in range(num_iters):
            if not freeze_top:
                delta = self.layers[0].backward(self.errs[1])
                self.mus[0] = self.mus[0] + self.mu_dt * (2 * delta)
            for l in range(1, self.num_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.deltas[l] = delta
                self.mus[l] = self.mus[l] + self.mu_dt * (2 * delta)

            for n in range(1, self.num_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

            avg_err = self.get_errors()[0] / self.total_params
            avg_errs.append(avg_err)

            avg_delta = self.get_deltas() / self.total_params
            avg_deltas.append(avg_delta)

            # get delta difference
            if itr > 0:
                if delta_thresh is not None and avg_delta < delta_thresh:
                    break

            if thresh is not None and avg_err < thresh:
                break

        return itr, avg_errs

    def test_updates(
        self, num_iters, fixed_preds=False, thresh=None, delta_thresh=None
    ):
        for n in range(1, self.num_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        avg_errs = []
        avg_err = self.get_errors()[0] / self.total_params
        avg_errs.append(avg_err)

        avg_deltas = []
        itr = 0
        for itr in range(num_iters):
            delta = self.layers[0].backward(self.errs[1])
            self.mus[0] = self.mus[0] + self.mu_dt * (2 * delta)
            for l in range(1, self.num_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.deltas[l] = delta
                self.mus[l] = self.mus[l] + self.mu_dt * (2 * delta)

            for n in range(1, self.num_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

            avg_err = self.get_errors()[0] / self.total_params
            avg_errs.append(avg_err)

            avg_delta = self.get_deltas() / self.total_params
            avg_deltas.append(avg_delta)

            if itr > 0:
                if delta_thresh is not None and avg_delta < delta_thresh:
                    break

            if thresh is not None and avg_err < thresh:
                break

        return itr, avg_errs

    def update_grads(self):
        for l in range(self.num_layers):
            self.layers[l].update_gradient(self.errs[l + 1])

    def update_amort_grads(self, net_i):
        for q, l in zip(reversed(range(1, self.num_nodes)), range(self.num_layers)):
            self.q_errs[q] = self.mus[l] - self.q_preds[q]

        for l in range(self.num_layers):
            for n in range(self.n_amort_nets):
                # determine the entries for this network in the batch
                batch_net = net_i == n
                self.amort_nets[n][l].update_gradient(self.q_errs[l + 1][batch_net])

    def get_errors(self):
        total_err = 0
        for err in self.errs:
            if len(err) > 0:
                total_err = total_err + torch.sum(torch.abs(err)).item()

        q_total_err = 0
        for err in self.q_errs:
            if len(err) > 0:
                q_total_err = q_total_err + torch.sum(torch.abs(err)).item()

        return total_err, q_total_err

    def get_deltas(self):
        total_delta = 0
        for delta in self.deltas:
            if len(delta) > 0:
                total_delta = total_delta + torch.sum(torch.abs(delta)).item()
        return total_delta

    def get_losses(self):
        try:
            return (
                torch.sum(torch.abs(self.errs[-1])).item(),
                torch.sum(torch.abs(self.q_errs[-1])).item(),
            )
        except:
            return torch.sum(torch.abs(self.errs[-1])).item(), 0

    def get_weight_stats(self):
        # fine here
        mean_abs_weights, mean_abs_biases = [], []
        for l in range(self.num_layers):
            mean_abs_weights.append(
                torch.mean(torch.abs(self.layers[l].weights)).item()
            )
            mean_abs_biases.append(torch.mean(torch.abs(self.layers[l].bias)).item())
        return mean_abs_weights, mean_abs_biases

    def normalize_weights(self):
        for l in range(self.num_layers):
            mean_weights = torch.mean(torch.abs(self.layers[l].weights))
            self.layers[l].weights = (
                self.layers[l].weights * self.mean_weights[l] / mean_weights
            )
            # VN: For some reason, this function had normalization of biases and causing a division by zero error.
            # I have chosen to keep it around, but only if the mean bias is greater than 0.
            if self.layers[l].use_bias and self.mean_biases[l] > 0:
                mean_bias = torch.mean(torch.abs(self.layers[l].bias))
                self.layers[l].bias = (
                    self.layers[l].bias * self.mean_biases[l] / mean_bias
                )

    @property
    def params(self):
        params = self.layers
        for net in self.amort_nets:
            params = params + net
        return params

    def __str__(self):
        return f"<DoubleAmortModelModel> {self.nodes}"
