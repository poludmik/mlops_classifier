import os
import random
import click
import torch
from torch import nn, optim
from models.model import MyAwesomeModel
import matplotlib.pyplot as plt

from data.make_dataset import mnist

import hydra


@hydra.main(config_path="", config_name="config_train.yaml", version_base="1.1")
def main(cfg):
    """Train a model on MNIST."""
    print("Training day and night")
    lr = cfg.hyperparameters.learning_rate
    print(lr)

    epochs = cfg.hyperparameters.epochs
    bs = cfg.hyperparameters.batch_size
    lr = cfg.hyperparameters.learning_rate
    epochs = cfg.hyperparameters.epochs
    random.seed(cfg.hyperparameters.seed)
    torch.manual_seed(cfg.hyperparameters.seed)

    # TODO: Implement training loop here

    model = MyAwesomeModel()
    train_set = torch.load("../../../data/processed/corruptmnist/train.pt")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []  # List to store train losses

    for epoch in range(epochs):
        train_loss = 0
        for images, labels in train_loader:
            # print(images.shape, labels.shape)
            # Move images and labels to GPU if available
            # images, labels = images.to(device), labels.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(images)
            loss = criterion(output, labels)

            # Backward pass and optimization
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

        train_losses.append(train_loss / len(train_loader))  # Compute average train loss
        print(f"Epoch {epoch+1}/{epochs}.. Train loss: {train_loss/len(train_loader):.3f}")

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Plotting the train loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("../../../reports/figures/MyAwesomeModelLoss.png")
    plt.show()

    accuracy = 100 * correct / total
    print(f"Accuracy on the training set: {round(accuracy, 4)}%")

    # Save the model
    torch.save(model, "../../../models/MyAwesomeModel/model.pt")

if __name__ == "__main__":
    main()