from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

import src.model.utils as mutils
from src.data.utils import load_json
from src.metadata import metadata

CPU_DEVICE = torch.device("cpu")
NUM_CLASSES = 2


def load_model_for_inference(checkpoint_path, idx_to_label_path):
    # get the checkpoint;
    ckpt = torch.load(checkpoint_path, map_location=CPU_DEVICE)
    # see if should load in bf16;
    is_bf16 = "bf16" in str(checkpoint_path).split("/")[-1]
    # get tokenizer;
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=metadata.SAVED_MODELS_DIR,
        use_fast=True,
    )
    # get sentence bert;
    sbert = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=metadata.SAVED_MODELS_DIR,
    )
    sbert.requires_grad_(False)
    if is_bf16:
        sbert = sbert.to(torch.bfloat16)

    # get hyper params;
    hparams = ckpt["hyper_parameters"]
    if hparams["do_lora"]:
        # load lora module list;
        lora_module_list = mutils.init_lora_module_list_qv(
            sbert,
            rank=hparams["lora_rank"],
            alpha=hparams["lora_alpha"],
            do_ffn=hparams["do_ffn"],
        )
        if is_bf16:
            lora_module_list = lora_module_list.to(torch.bfloat16)
        lora_module_list.load_state_dict(
            ckpt["state_dict"]["lora_module_list"]
        )
        # load lora modules into sbert;
        mutils.load_lora_layers_qv_(
            sbert=sbert,
            lora_layers=lora_module_list,
            do_ffn=hparams["do_ffn"],
        )
        # bind lora weights into sbert weights for fast inference;
        mutils.bind_lora_qv_(sbert, do_ffn=hparams["do_ffn"])

    # get d_model
    d_model = sbert.pooler.dense.in_features

    # load mlp;
    mlp = mutils.MLP(d_model, NUM_CLASSES)
    if is_bf16:
        mlp = mlp.to(torch.bfloat16)
    mlp.load_state_dict(ckpt["state_dict"]["mlp"])

    # read idx_to_label;
    idx_to_label = load_json(idx_to_label_path)

    # combine sbert and the mlp classifier;
    model = CombinedModel(
        sbert=sbert, mlp=mlp, tokenizer=tokenizer, idx_to_label=idx_to_label
    )
    return model


class CombinedModel(nn.Module):
    def __init__(self, sbert, mlp, tokenizer, idx_to_label):
        super().__init__()
        self.sbert = sbert
        self.mlp = mlp
        self.tokenizer = tokenizer
        self.idx_to_label = idx_to_label

    def _get_sentence_embed(self, sbert_output, attention_mask):
        # attention mask is (batch_size, seq_len) and has zeros
        # where tokens should be ignored, and ones otherwise.
        # to do the mean pooling I divide by num non_zero entries
        # for each sequence.
        attention_mask = attention_mask / attention_mask.sum(-1, keepdim=True)
        # sbert_output['last_hidden_state'] is (batch_size, seq_len, embed_dim)
        # transposing the last two dims gives (batch_size, embed_dim, seq_len)
        # then I do mean pooling by (batch) vector multiplying by the attentino_mask
        # this only selects embeds of relevant tokens since att_mask is zero otherwise.
        # resulting mean_pooled should be of shape (batch_size, embed_dim, 1)
        mean_pooled = sbert_output["last_hidden_state"].transpose(
            -1, -2
        ) @ attention_mask.unsqueeze(-1)
        # squeeze to get (batch_size, embed_dim)
        return mean_pooled.squeeze()

    def _cast_att_mask_to_dtype_(self, tokenizer_output):
        """
        Inplace change dtype of attention mask to that of self.mlp.
        """
        mem_format = next(self.mlp.parameters()).dtype
        att_mask = tokenizer_output["attention_mask"]
        tokenizer_output["attention_mask"] = att_mask.to(mem_format)

    def _forward_pass(
        self,
        strings,
    ):
        x = self.tokenizer(
            strings,
            padding="longest",
            return_tensors="pt",
            truncation="longest_first",
            max_length=256,
        )
        # cast dtype of tensors in x to that of self.mlp
        self._cast_att_mask_to_dtype_(x)
        device = next(self.mlp.parameters()).device
        x = x.to(device)
        sbert_out = self.sbert(**x)
        sent_embed = self._get_sentence_embed(sbert_out, x["attention_mask"])
        return self.mlp(sent_embed)

    def forward(self, strings):
        return self._forward_pass(strings)

    def get_predictions(self, strings):
        preds = self(strings).argmax(-1, keepdims=True)

        # get the labels;
        ans = []
        for p in preds:
            # if class is on, add the corresponding label
            # at that index;
            ans.append(self.idx_to_label[p.item()])
        return ans
