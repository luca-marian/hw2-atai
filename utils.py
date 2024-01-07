import re
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def predict(model, transform, img_path, options=None):
    img = Image.open(os.path.join(options.dataset_valid, img_path)).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(options.device)

    model.eval()

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

    return preds[0].item()


def write_submission(
    data: pd.DataFrame, task_name: str = None, filename: str = None, options=None
):
    submission_dir = os.path.join(options.dataset_root, "submission", task_name)

    if task_name not in ["task1", "task2"]:
        raise TypeError(
            "Task name is mandatory and can have the value: Task1 or Task2."
        )

    if not Path(submission_dir).exists():
        os.makedirs(submission_dir)

    if filename is None:
        filename = "submission"

    _filename = os.path.join(options.dataset_root, submission_dir, filename + ".csv")

    index = 1
    while os.path.exists(_filename):
        _filename = os.path.join(
            options.dataset_root, submission_dir, filename + "_" + str(index) + ".csv"
        )
        index = index + 1

    data.to_csv(_filename, index=False)


def create_data(dictionary: dict):
    df = pd.DataFrame.from_dict(dictionary, orient="index")
    df.reset_index(inplace=True)
    df.columns = ["sample", "label"]

    return df


def create_submission(model, transform=None, task_name: str = None, options=None):
    if options.dataset_valid is None:
        raise TypeError("The path must be str")

    image_paths = sorted(os.listdir(options.dataset_valid), key=natural_sort_key)

    valid_dict = {}
    for img_path in tqdm(image_paths, desc="Predict images"):
        valid_dict[img_path] = predict(model, transform, img_path, options)

    data = create_data(valid_dict)

    write_submission(data, task_name, options=options)


def create_combined_dataframe(data_path_1: str or Path, data_2: pd.DataFrame, options):
    df_labeled = pd.read_csv(data_path_1)

    print(type(df_labeled))

    df_labeled["sample"] = options.dataset_root + "/" + df_labeled["sample"]

    return pd.concat([df_labeled, data_2], ignore_index=True)
