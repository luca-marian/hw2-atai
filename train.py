import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision.io import read_image, ImageReadMode
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, Subset

from itertools import chain

import time
import os

from tempfile import TemporaryDirectory


# TASK1 - training loop
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    dataset_sizes,
    num_epochs=25,
    options=None,
):
    model.to(options.device)
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Training phase
            model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in train_loader:
                inputs = inputs.to(options.device)
                labels = labels.to(options.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step()

            epoch_loss = running_loss / dataset_sizes[0]
            epoch_acc = running_corrects.float() / dataset_sizes[0]

            print(f"Train\n\t Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            print()

            model.eval()

            running_loss = 0.0
            running_corrects = 0
            # Validation phase
            for inputs, labels in valid_loader:
                inputs = inputs.to(options.device)
                labels = labels.to(options.device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[1]
            epoch_acc = running_corrects.float() / dataset_sizes[1]

            print(f"Valid\n\t Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    torch.save(model.state_dict(), "best_noisy_model.py")

    return model


def evaluate_model(
    model, criterion, valid_loader, valid_size, title: str = "Model", options=None
):
    running_loss = 0.0
    running_corrects = 0
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(options.device), labels.to(options.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(predicted == labels.data)

    epoch_loss = running_loss / valid_size
    epoch_acc = running_corrects.float() / valid_size

    print(f"Valid {title}\n\t Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return epoch_acc


def generate_pseudo_labels(
    teacher_model, dataloader, confidences_threshold=0.8, options=None
):
    # Store pseudo-labeled data
    pseudo_labeled_data = []

    teacher_model.eval()
    with torch.no_grad():
        for images, _, img_paths in tqdm(dataloader, desc="Generate PSEUDO-LABELS"):
            images = images.to(options.device)
            outputs = teacher_model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, pseudo_labels = torch.max(probabilities, dim=1)

            # Filter out low confidence predictions
            high_confidence_mask = confidences > confidences_threshold

            high_confidence_images = images[high_confidence_mask]
            high_confidence_labels = pseudo_labels[high_confidence_mask]
            high_confidence_paths = [
                img_paths[i] for i in range(len(img_paths)) if high_confidence_mask[i]
            ]

            for img, path, label in zip(
                high_confidence_images, high_confidence_paths, high_confidence_labels
            ):
                pseudo_labeled_data.append((img.cpu(), path, label.cpu()))

    pseudo_labels_dict = {
        "sample": [row[1] for row in pseudo_labeled_data],
        "label": [row[2].item() for row in pseudo_labeled_data],
    }

    df_pseudo = pd.DataFrame(pseudo_labels_dict)

    print(f"Generated new {len(df_pseudo)} pseudo labels")

    return df_pseudo


def train_noisy_student(
    teacher,
    student,
    criterion_teacher,
    optimizer_teacher,
    scheduler_teacher,
    criterion_student,
    optimizer_student,
    scheduler_student,
    train_loader,
    valid_loader,
    unlabeled_loader,
    dataset_sizes,
    num_epochs=25,
    options=None,
):
    teacher.to(options.device)
    student.to(options.device)

    # Train the model
    best_acc = 0.0
    best_model = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        running_loss_teacher = 0.0
        running_corrects_teacher = 0

        student.train()
        teacher.train()

        for labeled_data, labels in train_loader:
            labeled_data = labeled_data.to(options.device)
            labels = labels.to(options.device)

            # Train on labeled data
            optimizer_teacher.zero_grad()
            outputs = teacher(labeled_data)
            _, preds = torch.max(outputs, 1)
            loss_teacher = criterion_teacher(outputs, labels)
            loss_teacher.backward()
            optimizer_teacher.step()

            teacher.eval()
            # Generate pseudo-labels for unlabeled data
            with torch.no_grad():
                unlabeled_data, _, _ = next(iter(unlabeled_loader))
                unlabeled_data = unlabeled_data.to(options.device)
                outputs = teacher(unlabeled_data)
                confidences, pseudo_labels = torch.max(outputs, 1)

                high_confidence_mask = confidences > 0.95
                unlabeled_data = unlabeled_data[high_confidence_mask]
                pseudo_labels = pseudo_labels[high_confidence_mask]

            # Train on labeled and pseudo-labeled data
            # Student model
            optimizer_student.zero_grad()
            labeled_outputs = student(labeled_data)
            unlabeled_outputs = student(unlabeled_data)
            loss_student = criterion_student(
                labeled_outputs, labels
            ) + criterion_student(unlabeled_outputs, pseudo_labels)
            loss_student.backward()
            optimizer_student.step()

            running_loss_teacher += loss_teacher.item() * labeled_data.size(0)
            running_corrects_teacher += torch.sum(preds == labels.data)

        scheduler_teacher.step()
        scheduler_student.step()

        epoch_loss = running_loss_teacher / dataset_sizes[0]
        epoch_acc = running_corrects_teacher.float() / dataset_sizes[0]

        print(f"Train\n\t Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        teacher_acc = evaluate_model(
            teacher,
            criterion_teacher,
            valid_loader,
            dataset_sizes[1],
            "Teacher",
            options,
        )
        student_acc = evaluate_model(
            student,
            criterion_teacher,
            valid_loader,
            dataset_sizes[1],
            "Student",
            options,
        )

        if teacher_acc > best_acc:
            best_acc = teacher_acc
            best_model = teacher
            torch.save(teacher.state_dict(), "best_noisy_model.pt")
        elif student_acc > best_acc:
            best_acc = student_acc
            best_model = student
            torch.save(student.state_dict(), "best_noisy_model.pt")

    best_model.load_state_dict(torch.load("best_noisy_model.pt"))

    return best_model


# Task2 - training loop
def co_teaching_loss(model1_loss, model2_loss, rt, options):
    _, model1_sm_idx = torch.topk(
        model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False
    )
    _, model2_sm_idx = torch.topk(
        model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False
    )

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).to(options.device)
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss = (model1_loss_filter * model1_loss).sum()

    model2_loss_filter = torch.zeros((model2_loss.size(0))).to(options.device)
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).sum()

    return model1_loss, model2_loss


def train_step(data_loader, model_list: list, optimizer, criterion, rt, options):
    global_step = 0
    avg_accuracy = 0.0
    avg_loss = 0.0

    model1, model2 = model_list
    model1 = model1.train()
    model2 = model2.train()
    for x, y in data_loader:
        # Forward and Backward propagation
        x, y, y_hat = (
            x.to(options.device),
            y.to(options.device),
        )

        out1 = model1(x)
        out2 = model2(x)

        model1_loss = criterion(out1, y_hat)
        model2_loss = criterion(out2, y_hat)
        model1_loss, model2_loss = co_teaching_loss(
            model1_loss=model1_loss, model2_loss=model2_loss, rt=rt, options=options
        )

        # loss exchange
        optimizer.zero_grad()
        model1_loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 5.0)
        optimizer.step()

        optimizer.zero_grad()
        model2_loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
        optimizer.step()

        avg_loss += model1_loss.item() + model2_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out1, 1), y).float()
        avg_accuracy += acc.mean()
        global_step += 1

    return avg_accuracy / global_step, avg_loss / global_step, [model1, model2]


def test_step(data_loader, model, options):
    model = model.eval()
    global_step = 0
    avg_accuracy = 0.0

    for x, y in data_loader:
        x, y = x.to(options.device), y.to(options.device)

        logits = model(x)
        acc = torch.eq(torch.argmax(logits, 1), y)
        acc = acc.cpu().numpy()
        acc = np.mean(acc)
        avg_accuracy += acc
        global_step += 1
    return avg_accuracy / global_step


def update_reduce_step(cur_step, num_gradual, tau=0.5):
    return 1.0 - tau * min(cur_step / num_gradual, 1)


def train_co_teaching(
    train_data_loader,
    valid_data_loader,
    tau=0.5,
    num_gradual=15,
    num_epochs=10,
    lr=0.001,
    model1=None,
    model2=None,
    options=None,
):
    if model1 == None:
        model1 = models.resnet50()

    if model2 == None:
        model2 = models.resnet50()

    model1.to(options.device)
    model2.to(options.device)

    # learning history
    train_acc_list = []
    test_acc_list = []

    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(chain(model1.parameters(), model2.parameters()), lr=lr)

    for e in range(num_epochs):
        # update reduce step
        rt = update_reduce_step(cur_step=e, num_gradual=num_gradual, tau=tau)

        # training step
        train_accuracy, avg_loss, model_list = train_step(
            data_loader=train_data_loader,
            model_list=[model1, model2],
            optimizer=optimizer,
            criterion=criterion,
            rt=rt,
            options=options,
        )
        model1, model2 = model_list

        # testing/valid step
        test_accuracy = test_step(
            data_loader=valid_data_loader, model=model1, options=options
        )

        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

        print(
            "{} epoch, Train Loss {}, Train accuracy {}, Test accuracy {}, Reduce rate {}".format(
                e + 1, avg_loss, train_accuracy, test_accuracy, rt
            )
        )

    return model1, model2
