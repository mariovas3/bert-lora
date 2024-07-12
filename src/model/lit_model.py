from itertools import chain
from pathlib import Path

import torch
from lightning import LightningModule
from torch import nn
from transformers import AutoModel, AutoTokenizer

from src.model import utils

MODEL_CACHE_DIR = Path(__file__).absolute().parents[2] / "saved_models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class LitModel(LightningModule):
    def __init__(
        self,
        idx_to_label: list[str],
        lora_rank: int = 8,
        lora_alpha: float = 1,
        lr: float = 1e-3,
        do_ffn: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lr = lr
        self.idx_to_label = idx_to_label
        # whether to do lora weights on the mlp
        # of the encoder block;
        self.do_ffn = do_ffn
        # get model and tokenizer from transformers;
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=MODEL_CACHE_DIR,
            use_fast=True,
        )
        self.sbert = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=MODEL_CACHE_DIR,
        )
        # stop grads for sbert;
        self.sbert.requires_grad_(False)
        self.lora_module_list = utils.init_lora_module_list_qv(
            self.sbert,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            do_ffn=self.do_ffn,
        )
        # inplace make qv and possibly ffn's linear layers
        # into lora layers;
        utils.load_lora_layers_qv_(
            sbert=self.sbert,
            lora_layers=self.lora_module_list,
            do_ffn=self.do_ffn,
        )
        self.d_model = self.sbert.pooler.dense.in_features
        self.num_classes = len(idx_to_label)
        self.mlp = utils.MLP(self.d_model, self.num_classes)

    def configure_optimizers(self):
        return torch.optim.Adam(
            chain(self.lora_module_list.parameters(), self.mlp.parameters()),
            lr=self.lr,
        )

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

    def _forward_pass(self, strings):
        x = self.tokenizer(
            strings,
            padding="longest",
            return_tensors="pt",
            truncation="longest_first",
            max_length=256,
        )
        x = x.to(self.device)
        sbert_out = self.sbert(**x)
        sent_embed = self._get_sentence_embed(sbert_out, x["attention_mask"])
        return self.mlp(sent_embed)

    def forward(self, strings):
        return self._forward_pass(strings)

    def _get_loss(self, logits, targets):
        return nn.functional.cross_entropy(logits, targets)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        strings, targets = batch
        logits = self(strings)
        loss = self._get_loss(logits, targets)

        # log to logger;
        self.log(
            "training/loss",
            loss.item(),
            logger=True,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        strings, targets = batch
        logits = self(strings)
        loss = self._get_loss(logits, targets).item()

        # get accuracy;
        acc = (logits.argmax(-1) == targets).float().mean().item()

        self.log_dict(
            {
                "validation/loss": loss,
                "validation/accuracy": acc,
            },
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )

    def predict_step(self, batch, batch_idx) -> list[list[str]]:
        # expect not to have targets;
        strings = batch
        # see which classes are "on"
        preds = self(strings).argmax(-1)

        # get the labels;
        ans = []
        for p in preds:
            # if class is on, add the corresponding label
            # at that index;
            ans.append(self.idx_to_label[p.item()])
        return ans

    def on_save_checkpoint(self, checkpoint):
        lora_module_list = utils.extract_lora_layers_qv(
            sbert=self.sbert, do_ffn=self.do_ffn
        )
        checkpoint["state_dict"] = dict(
            lora_module_list=lora_module_list.state_dict(),
            mlp=self.mlp.state_dict(),
        )
