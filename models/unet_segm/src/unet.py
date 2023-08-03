import torch
import torch.nn as nn
import wandb
from meddlr.ops import complex as cplx
import numpy as np
import matplotlib.pyplot as plt
from models.unet_segm.src.unet_parts import *


# U-Net architecture
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def loss(self, pred, target):
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        loss = torch.mean(
            torch.sum(criterion(input=pred, target=target.long()), dim=(1, 2))
        )
        return loss

    def make_prediction(self, img):
        """Performs a prediction given a reconstructed image

        Args:
            img (torch.Tensor): The reconstructed image

        Returns:
            torch.Tensor: The segmentation prediction
        """
        out = self.forward(img)
        out_pred = torch.argmax(out, dim=1).squeeze()
        return out_pred


def compute_train_loss_and_train(train_loader, model, optimizer, use_gpu, epoch):
    model.train()

    running_loss = 0.0

    for x, y, _, _ in train_loader:
        if use_gpu:
            x = x.cuda()
            y = y.cuda()

        # compute forward pass
        output = model(x)

        loss = model.loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss * train_loader.batch_size
        torch.cuda.empty_cache()

    epoch_loss = running_loss / len(train_loader.dataset)

    return epoch_loss


def compute_eval_loss(test_loader, model, use_gpu, epoch):
    """
    computes the evaluation epoch loss on the evaluation set
    """
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for x, y, _, _ in test_loader:
            if use_gpu:
                x = x.cuda()
                y = y.cuda()

            # compute forward pass
            output = model(x)

            loss = model.loss(output, y)

            running_loss += loss * test_loader.batch_size
    torch.cuda.empty_cache()

    epoch_loss = running_loss / len(test_loader.dataset)
    return epoch_loss


def train_model(
    model,
    train_loader,
    eval_loader,
    optim,
    epochs=1,
    save_model=None,
    save_path=None,
    continue_training_path=None,
    eval_metric=None,
):
    end_epoch = 0
    use_gpu = torch.cuda.is_available()

    if continue_training_path:
        checkpoint = torch.load(continue_training_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if use_gpu:
            model.cuda()
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        end_epoch = checkpoint["epoch"]
    if use_gpu:
        model.cuda()

    # define current best losses
    best_total_eval_loss = np.inf

    for epoch in range(end_epoch, epochs):
        print("Epoch:", epoch)

        # train the model
        train_running_loss = compute_train_loss_and_train(
            train_loader, model, optim, use_gpu, epoch=epoch
        )

        # compute evaluation loss
        eval_running_loss = compute_eval_loss(eval_loader, model, use_gpu, epoch)

        if save_model == True:
            if eval_running_loss < best_total_eval_loss:
                best_total_eval_loss = eval_running_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "loss": train_running_loss,
                    },
                    f"{save_path}unet_seg_best_eval_epoch{epoch}.pth",
                )
                print("saving best eval model")

        print("training loss:", train_running_loss)
        print("evaluation loss:", eval_running_loss)

    return
