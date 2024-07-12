from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from src.data.lit_data import LitData
from src.model.lit_model import LitModel


class MyLitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "trainer.accelerator",
            "data.on_gpu",
            compute_fn=lambda x: x == "gpu",
            apply_on="parse",
        )
        # make ModelCheckpoint callback configurable;
        parser.add_lightning_class_args(ModelCheckpoint, "my_model_checkpoint")
        parser.set_defaults(
            {
                "my_model_checkpoint.monitor": "validation/loss",
                "my_model_checkpoint.mode": "min",
                "my_model_checkpoint.every_n_epochs": 2,
            }
        )


def main():
    cli = MyLitCLI(
        model_class=LitModel,
        datamodule_class=LitData,
        seed_everything_default=0,
    )


if __name__ == "__main__":
    main()
