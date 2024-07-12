import json
import re
from math import sqrt
from pathlib import Path

from src.metadata import metadata


def save_to_json(obj, filepath: Path, **kwargs):
    parent_dir = filepath.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as file:
        json.dump(obj, file, **kwargs)


def load_json(filepath: Path):
    with open(filepath, "r") as file:
        obj = json.load(file)
    return obj


def get_idx_to_label(targets: list[tuple[str]]) -> list[str]:
    return sorted(set([label for labels in targets for label in labels]))


def get_dict_from_dataset(dataset):
    texts, targets = [], []
    for record in dataset:
        texts.append(clean_text(record["text"]))
        targets.append(record["label"])
    return {"texts": texts, "targets": targets}


def process_and_save(dataset, processed_dir: Path):
    train_dict = get_dict_from_dataset(dataset["train"])
    test_dict = get_dict_from_dataset(dataset["test"])
    save_to_json(train_dict, processed_dir / "train.json")
    save_to_json(test_dict, processed_dir / "val.json")
    config = {
        "train": {
            "features": ["texts", "targets"],
            "num_samples": len(train_dict["targets"]),
        },
        "test": {
            "features": ["texts", "targets"],
            "num_samples": len(test_dict["targets"]),
        },
    }
    save_to_json(config, metadata.METADATA_JSON_PATH, indent=2)


def clean_text(text):
    # remove space + newline combo to just space;
    text = re.sub("\s*\n+", " ", text.strip())
    # remove quotes and sub multiple spaces for single space
    # also strip commas;
    text = re.sub("\s+", " ", re.sub('"', "", text)).strip(",")
    return re.sub("(<br />)+", " ", text)


def get_min_max_mean_std(texts: list[str]):
    """
    Returns min, max, mean, std of lens of split elements of texts.
    """
    n = 0
    avg = 0
    avg_of_square = 0
    curr_min, curr_max = 1e8, -1
    for text in texts:
        l = len(text.split())
        curr_min = min(curr_min, l)
        curr_max = max(curr_max, l)
        n += 1
        avg = avg + (l - avg) / n
        avg_of_square = avg_of_square + (l * l - avg_of_square) / n
    std = sqrt(avg_of_square - avg * avg)
    return curr_min, curr_max, avg, std
