import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def get_model(pretrained_name="resnet50"):
    if pretrained_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
    elif pretrained_name == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V2")
    elif pretrained_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V2")
    else:
        raise TypeError("You can choose only 3 models: resnet50, resnet34, resnet18 ")

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 100)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, exp_lr_scheduler


class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB, modelC, options):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA.to(options.device)
        self.modelB = modelB.to(options.device)
        self.modelC = modelC.to(options.device)
        # You can also add fully connected layers here if you want to combine features

    def forward(self, x):
        # Assume that each model outputs log probabilities
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)

        # Average the predictions
        x = torch.mean(torch.stack([x1, x2, x3]), dim=0)
        return x
