from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import src.metadata.metadata as metadata
from src.data import utils


class LitData(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 50,
        num_workers: int = 1,
        on_gpu: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.on_gpu = on_gpu
        self.seed = seed
        self.raw_data_dir = metadata.RAW_DATA_DIR
        self.processed_data_dir = metadata.PROCESSED_DATA_DIR

    def prepare_data(self):
        if metadata.IDX_TO_LABEL_PATH.exists():
            print(f"DATA ALREADY PREPARED!")
            self.idx_to_label = utils.load_json(metadata.IDX_TO_LABEL_PATH)
            self.label_to_idx = {
                label: i for i, label in enumerate(self.idx_to_label)
            }
            self.num_classes = len(self.idx_to_label)
            return
        print(f"PREPARING DATA...")
        dataset = load_dataset("imdb", cache_dir=metadata.RAW_DATA_DIR)
        utils.process_and_save(dataset, self.processed_data_dir)
        self.idx_to_label = ["negative", "positive"]
        self.label_to_idx = {
            label: i for i, label in enumerate(self.idx_to_label)
        }
        utils.save_to_json(
            self.idx_to_label, metadata.IDX_TO_LABEL_PATH, indent=2
        )
        utils.save_to_json(
            self.idx_to_label, metadata.SAVED_MODELS_DIR / "idx_to_label.json"
        )
        print(f"DATA PREP DONE!")

    def __repr__(self):
        config_dict = utils.load_json(metadata.METADATA_JSON_PATH)
        return repr(config_dict)

    def setup(self, stage: str):
        if stage != "fit":
            raise ValueError(f"stage must be fit but {stage} given")
        train = utils.load_json(self.processed_data_dir / "train.json")
        val = utils.load_json(self.processed_data_dir / "val.json")
        self.train_dataset = MyDataset(x=train["texts"], y=train["targets"])
        self.val_dataset = MyDataset(x=val["texts"], y=val["targets"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )


class MyDataset(Dataset):
    def __init__(self, x, y=None):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        return self.x[index], self.y[index]


if __name__ == "__main__":
    import torch

    # quick mock;
    data = LitData(
        batch_size=50,
        num_workers=1,
        gpus=0,
        seed=0,
    )

    # prepare data;
    data.prepare_data()
    info = utils.load_json(metadata.METADATA_JSON_PATH)
    train_processed_len = info["train"]["num_samples"]
    val_processed_len = info["test"]["num_samples"]
    print(data)

    # check the fit stage;
    data.setup("fit")
    train_loader, val_loader = data.train_dataloader(), data.val_dataloader()
    assert len(train_loader.dataset) == train_processed_len
    assert len(val_loader.dataset) == val_processed_len
    x_train, y_train = next(iter(train_loader))
    assert torch.all((y_train == 0) | (y_train == 1))
    x_val, y_val = next(iter(val_loader))
    assert torch.all((y_val == 0) | (y_val == 1))
