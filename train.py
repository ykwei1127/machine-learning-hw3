import torch
import torch.nn as nn
from datasets.dataloader import make_train_dataloader
from torchvision.models.resnet import resnet18
from models.model import MyResnet18

import os
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 50
learning_rate = 0.001

base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "data", "train")
weight_path = os.path.join(base_path, "weights", "weight.pth")

# make dataloader for train data
train_loader, valid_loader = make_train_dataloader(train_data_path)

# set model
resnet = resnet18(pretrained=False)
model = MyResnet18(resnet)
model = model.to(device)

# set optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_loss_list = list()
valid_loss_list = list()
train_accuracy_list = list()
for epoch in range(epochs):
    print(f'\nEpoch: {epoch+1}/{epochs}')
    print('-' * len(f'Epoch: {epoch+1}/{epochs}'))
    train_loss = 0.0
    valid_loss = 0.0
    predict_correct = 0
    train_accuracy = 0.0
    best_loss = 100

    model.train()
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)

        # forward + backward + optimize
        output  = model(data)
        _, preds = torch.max(output.data, 1)
        loss = criterion(output, target)
        optimizer.zero_grad()   # zero the parameter gradients
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        predict_correct += torch.sum(preds == target.data)

    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_accuracy = float(predict_correct) / len(train_loader.dataset)
    train_accuracy_list.append((train_accuracy))

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(valid_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            valid_loss += loss.item() * data.size(0)
        valid_loss /= len(valid_loader.dataset)
        valid_loss_list.append(valid_loss)
    
    # print loss in one epoch
    print(f'Training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}')

    if valid_loss < best_loss :
            best = valid_loss
            torch.save(model.state_dict(), weight_path)

print("\nFinished Training")
pd.DataFrame({
    "train-loss": train_loss_list,
    "valid-loss": valid_loss_list
}).plot()
plt.savefig(os.path.join(base_path, "result", "Loss_curve"))

pd.DataFrame({
    "train-accuracy": train_accuracy_list
}).plot()
plt.savefig(os.path.join(base_path, "result", "Training_accuracy"))