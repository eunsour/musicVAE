import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import model
import preprocess
import dataloader

from utils import *
from config import *

is_training_from_checkpoint = False


def save_checkpoint(
    path,
    model,
    optimizer,
    valid_acc,
    epoch,
    optimizer_scheduler,  # optinal
):
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "valid_acc": valid_acc,
        "epoch": epoch,
        "optimizer_scheduler_state_dict": optimizer_scheduler.state_dict(),
    }

    torch.save(save_dict, path)
    print("model checkpoint saved")
    return True


def load_checkpoint(path=params["path_model_trained"]):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        return checkpoint
    else:
        raise Exception("doest not exist checkpiont file")


if __name__ == "__main__":
    train_epochs = 100
    train_set_loader, valid_set_loader = dataloader.main()

    device = get_device()
    model = model.Model(device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    # continual training from check point
    if is_training_from_checkpoint:
        checkpoint = load_checkpoint()
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        previous_acc = checkpoint["valid_acc"]
        optimizer_scheduler = optimizer_scheduler.load_state_dict(checkpoint["optimizer_scheduler_state_dict"])

        print("start training from trained weight and optimizer state")

    else:
        start_epoch = 0
        previous_valid_acc = 0
        print("start training from zero weight")

    for epoch in range(start_epoch, train_epochs):
        train_loss, train_acc, valid_loss, valid_acc = 0, 0, 0, 0

        ## Train
        model.train()
        for batch_idx, train_set in enumerate(train_set_loader):
            train_set = train_set.to(device)

            batch_size = train_set.shape[0]
            seq_len = train_set.shape[1]

            optimizer.zero_grad()
            output_prob, mu, std = model(train_set)
            label = torch.argmax(output_prob, 2)

            # teacher forcing

            # loss
            beta = kl_annealing(epoch, 0, 0.2)
            loss = vae_loss(output_prob, train_set, mu, std, beta)

            # backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(train_set, label).item()

        train_loss = train_loss / (batch_idx + 1)
        train_acc = train_acc / (batch_idx + 1)

        optimizer_scheduler.step()

        ## Validation
        model.eval()  # turn off useless layer components of Model for inference
        with torch.no_grad():  # disable gradient calculation
            for batch_idx, valid_set in enumerate(valid_set_loader):
                valid_set = valid_set.to(device)

                output_prob, mu, std = model(valid_set)

                label = torch.argmax(output_prob, 2)

                loss = vae_loss(output_prob, valid_set, mu, std)
                valid_loss += loss.item()
                valid_acc += accuracy(valid_set, label).item()

            valid_loss = valid_loss / (batch_idx + 1)
            valid_acc = valid_acc / (batch_idx + 1)

        print(
            f"""
        train loss : {train_loss}, train_acc : {train_acc}, valid loss : {valid_loss}, valid_acc: {valid_acc}
        """
        )

        if epoch % 10 == 0:
            if previous_valid_acc < valid_acc:
                previous_valid_acc = valid_acc

                save_checkpoint(
                    path=params["path_model_trained"],
                    model=model,
                    optimizer=optimizer,
                    valid_acc=valid_acc,
                    epoch=epoch,
                    optimizer_scheduler=optimizer_scheduler,
                )
