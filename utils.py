import re
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import pandas as pd


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def predict(model, transform, img_path, base_dir=""):
    img = Image.open(os.path.join(base_dir, img_path)).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

    return preds[0].item()


def write_submission(data: pd.DataFrame, task_name: str = None, filename: str = None):
    submission_dir = os.path.join(DATA_PATH, "submission", task_name)

    if task_name not in ["task1", "task2"]:
        raise TypeError(
            "Task name is mandatory and can have the value: Task1 or Task2."
        )

    if not Path(submission_dir).exists():
        os.makedirs(submission_dir)

    if filename is None:
        filename = "submission"

    _filename = os.path.join(DATA_PATH, submission_dir, filename + ".csv")

    index = 1
    while os.path.exists(_filename):
        _filename = os.path.join(
            DATA_PATH, submission_dir, filename + "_" + str(index) + ".csv"
        )
        index = index + 1

    data.to_csv(_filename, index=False)


def create_data(dictionary: dict):
    df = pd.DataFrame.from_dict(dictionary, orient="index")
    df.reset_index(inplace=True)
    df.columns = ["sample", "label"]

    return df


def create_submission(
    model, transform=None, task_name: str = None, valid_dir: str = None
):
    if valid_dir is None:
        raise TypeError("The path must be str")

    image_paths = sorted(os.listdir(valid_dir), key=natural_sort_key)

    valid_dict = {}
    for img_path in tqdm(image_paths, desc="Predict images"):
        valid_dict[img_path] = predict(model, transform, img_path, valid_dir)

    data = create_data(valid_dict)

    write_submission(data, task_name)
