import torch
from torch import nn


def load_lora_layers_qv_(sbert, lora_layers: nn.ModuleList, do_ffn=False):
    """
    Inplace modify query and value layers, also ffn weights if do_ffn is True.

    Assumes lora_layers is list of [query_layer, value_layer] if do_ffn is False
    or [query_layer, value_layer, intermediate_ffn, output_ffn] if do_ffn is True.
    """
    mult = 4 if do_ffn else 2
    for i, layer in enumerate(sbert.encoder.layer):
        layer.attention.self.query = LinearWithLoRA.from_lora_layer(
            layer.attention.self.query, lora_layers[mult * i]
        )
        layer.attention.self.value = LinearWithLoRA.from_lora_layer(
            layer.attention.self.value, lora_layers[mult * i + 1]
        )
        if do_ffn:
            layer.intermediate.dense = LinearWithLoRA.from_lora_layer(
                layer.intermediate.dense, lora_layers[mult * i + 2]
            )
            layer.output.dense = LinearWithLoRA.from_lora_layer(
                layer.output.dense, lora_layers[mult * i + 3]
            )


def extract_lora_layers_qv(sbert, do_ffn=False):
    """
    Return nn.ModuleList of LoRA layers.

    Assumes lora layers only in query and value if do_ffn is False
    and also considers ffn's intermediate and output if do_ffn is True.
    """
    ml = nn.ModuleList()
    for layer in sbert.encoder.layer:
        ml.append(layer.attention.self.query.lora)
        ml.append(layer.attention.self.value.lora)
        if do_ffn:
            ml.append(layer.intermediate.dense.lora)
            ml.append(layer.output.dense.lora)
    return ml


def bind_lora_qv_(sbert, do_ffn=False):
    """
    Adds the lora approximation to the frozen weights inplace.

    This is for faster inference. Only query and value weights
    are considered if do_ffn is False, otherwise also consider
    ffn's intermediate and output weights.
    """
    for layer in sbert.encoder.layer:
        query = layer.attention.self.query
        if hasattr(query, "lora"):
            query.bind()
        value = layer.attention.self.value
        if hasattr(value, "lora"):
            value.bind()
        if do_ffn:
            intermediate = layer.intermediate.dense
            if hasattr(intermediate, "lora"):
                intermediate.bind()
            output = layer.output.dense
            if hasattr(output, "lora"):
                output.bind()


def unbind_lora_qv_(sbert, do_ffn=False):
    """
    Subtract the lora weights from the frozen weights inplace.

    Only query and value weights are considered if do_ffn is False,
    otherwise also consider ffn's intermediate and output weights.
    """
    for layer in sbert.encoder.layer:
        query = layer.attention.self.query
        if hasattr(query, "lora"):
            query.unbind()
        value = layer.attention.self.value
        if hasattr(value, "lora"):
            value.unbind()
        if do_ffn:
            intermediate = layer.intermediate.dense
            if hasattr(intermediate, "lora"):
                intermediate.unbind()
            output = layer.output.dense
            if hasattr(output, "lora"):
                output.unbind()


def init_lora_module_list_qv(sbert, rank, alpha, do_ffn=False):
    """
    Initialise nn.ModuleList with lora weights.

    Only init for query and value weights if do_ffn is False,
    otherwise also init for ffn's intermediate and output weights.
    """
    d_model = sbert.pooler.dense.in_features
    ffn_hidden = sbert.encoder.layer[0].intermediate.dense.out_features
    n_layers = len(sbert.encoder.layer)
    lora_weights = nn.ModuleList()
    for i in range(n_layers):
        lora_weights.append(LoRALayer(d_model, d_model, rank, alpha))
        lora_weights.append(LoRALayer(d_model, d_model, rank, alpha))
        if do_ffn:
            lora_weights.append(LoRALayer(d_model, ffn_hidden, rank, alpha))
            lora_weights.append(LoRALayer(ffn_hidden, d_model, rank, alpha))
    return lora_weights


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # expected in_dim = 384;
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.recip_rank, self.alpha = 1 / rank, alpha
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros((rank, out_dim)))

    def get_lora_approx(self):
        return self.A @ self.B * self.alpha * self.recip_rank

    def forward(self, x):
        return x @ self.get_lora_approx()


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            in_dim=linear.in_features,
            out_dim=linear.out_features,
            rank=rank,
            alpha=alpha,
        )
        self.bound = False

    @classmethod
    def from_lora_layer(cls, linear, lora_layer):
        ob = cls(linear, 1, 1)
        ob.lora = lora_layer
        return ob

    def bind(self):
        self.bound = True
        self.linear.weight.data = (
            self.linear.weight.data
            + self.lora.get_lora_approx().transpose(-1, -2)
        )

    def unbind(self):
        self.bound = False
        self.linear.weight.data = (
            self.linear.weight.data
            - self.lora.get_lora_approx().transpose(-1, -2)
        )

    def forward(self, x):
        if self.bound:
            return self.linear(x)
        return self.linear(x) + self.lora(x)
