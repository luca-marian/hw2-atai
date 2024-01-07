# Global imports
import os
import argparse, sys
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch

# Local imports
from data import LabeledData, UnlabeledData, get_split_dataset
import train
import models
import utils

DATA_PATH = os.path.join(os.getcwd(), "data")

TASK1_PATH = os.path.join(DATA_PATH, "task1")
TRAIN_TASK1 = os.path.join(TASK1_PATH, "train_data", "annotations.csv")
TRAIN_TASK1_UNLABELED = os.path.join(TASK1_PATH, "train_data", "images", "unlabeled")

VAL_TASK1 = os.path.join(TASK1_PATH, "val_data")

TASK2_PATH = os.path.join(DATA_PATH, "task2")
TRAIN_TASK2 = os.path.join(TASK2_PATH, "train_data", "annotations.csv")
VAL_TASK2 = os.path.join(TASK2_PATH, "val_data")


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, help="path to dataset", default="")
    parser.add_argument(
        "--result_dir",
        type=str,
        help="dir to save result csv files",
        default="submission/",
    )
    parser.add_argument(
        "--task_name", type=str, help="name of the task", default="task1"
    )
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--n_epoch", type=int, default=15)

    return parser.parse_args()


def get_device():
    # Check that MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device {device} selected")

    return device


normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)


def train_task1(options):
    if options.dataset_root == "":
        options.dataset_root = DATA_PATH
        options.dataset = TASK1_PATH
    else:
        options.dataset = os.path.join(options.dataset_root, "task1")

    options.dataset_train = os.path.join(
        options.dataset, "train_data", "annotations.csv"
    )

    options.dataset_unlabeled = os.path.join(
        options.dataset, "train_data", "images", "unlabeled"
    )

    options.dataset_valid = os.path.join(options.dataset, "val_data")

    (
        model_teacher,
        criterion_teacher,
        optimizer_teacher,
        scheduler_teacher,
    ) = models.get_model()

    dataset = LabeledData(options.dataset_train, transform, options)

    dataset_unlabeled = UnlabeledData(options.dataset_unlabeled, transform)

    unlabeled_loader = DataLoader(
        dataset_unlabeled, batch_size=options.batch_size, shuffle=True
    )

    train_loader, valid_loader, train_size, valid_size = get_split_dataset(
        dataset, batch_size=options.batch_size
    )

    print("Train baseline model")
    model_teacher = train.train_model(
        model_teacher,
        criterion_teacher,
        optimizer_teacher,
        scheduler_teacher,
        train_loader,
        valid_loader,
        (train_size, valid_size),
        options.n_epoch,
        options,
    )

    # Create submission file with simple model
    utils.create_submission(model_teacher, transform, options.task_name, options)

    (
        model_student,
        criterion_student,
        optimizer_student,
        scheduler_student,
    ) = models.get_model()

    print("Train noisy student with baseline model and another resnet50 model")
    model = train.train_noisy_student(
        model_teacher,
        model_student,
        criterion_teacher,
        optimizer_teacher,
        scheduler_teacher,
        criterion_student,
        optimizer_student,
        scheduler_student,
        train_loader,
        valid_loader,
        unlabeled_loader,
        (train_size, valid_size),
        options.n_epoch,
        options,
    )

    utils.create_submission(model, transform, options.task_name, options)

    print(
        f"Finish! The 2 submission file can be founded in submission folder on the root data {options.dataset_root}"
    )


def train_task2(options):
    if options.dataset_root == "":
        options.dataset_root = DATA_PATH
        options.dataset = TASK2_PATH
    else:
        options.dataset = os.path.join(options.dataset_root, "task2")

    options.dataset_train = os.path.join(
        options.dataset, "train_data", "annotations.csv"
    )

    options.dataset_valid = os.path.join(options.dataset, "val_data")

    dataset = LabeledData(options.dataset_train, transform, options)

    train_loader, valid_loader, train_size, valid_size = get_split_dataset(
        dataset, batch_size=options.batch_size
    )

    (
        model_teacher,
        criterion_teacher,
        optimizer_teacher,
        scheduler_teacher,
    ) = models.get_model()

    print("Train baseline model")
    model_teacher = train.train_model(
        model_teacher,
        criterion_teacher,
        optimizer_teacher,
        scheduler_teacher,
        train_loader,
        valid_loader,
        (train_size, valid_size),
        options.n_epoch,
        options,
    )

    # Create submission file with simple model
    utils.create_submission(model_teacher, transform, options.task_name, options)

    (
        model_student,
        criterion_student,
        optimizer_student,
        scheduler_student,
    ) = models.get_model()

    print("Train co-teaching model")
    m1, m2 = train.train_co_teaching(
        train_loader, valid_loader, num_epochs=options.n_epoch, options=options
    )

    utils.create_submission(m1, transform, options.task_name, options)
    utils.create_submission(m2, transform, options.task_name, options)


def main(options):
    print(options)

    options.device = get_device()

    if options.task_name not in ["task1", "task2"]:
        raise TypeError("Task name is invalid. You should type just task1 or task2")
    elif options.task_name == "task1":
        train_task1(options)

    elif options.task_name == "task2":
        train_task2(options)


if __name__ == "__main__":
    main(get_options())
