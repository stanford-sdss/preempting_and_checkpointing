import torch
import torch.nn as nn
import torch.nn.functional as F


# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_one_epoch(model, loss_fn, epoch_index, tb_writer, training_loader, optimizer,
                    stop_requested=lambda: False, report_every=500):
    """Train for one epoch. Returns the average loss over the batches run.

    stop_requested: callable checked before every batch; if it returns True
    (e.g. SIGTERM was received) we stop early so the caller can checkpoint.
    """
    running_loss = 0.
    total_loss = 0.
    n_batches = 0

    for i, (inputs, labels) in enumerate(training_loader):
        if stop_requested():
            print(f"  stopping early at batch {i} of {len(training_loader)}", flush=True)
            break

        optimizer.zero_grad()            # zero gradients for every batch
        outputs = model(inputs)          # forward pass
        loss = loss_fn(outputs, labels)  # loss and gradients
        loss.backward()
        optimizer.step()                 # update weights

        running_loss += loss.item()
        total_loss += loss.item()
        n_batches += 1
        if i % report_every == report_every - 1:
            last_loss = running_loss / report_every
            print(f"  batch {i + 1}/{len(training_loader)} loss: {last_loss:.4f}", flush=True)
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.

    return total_loss / max(n_batches, 1)


def evaluate(model, loss_fn, validation_loader):
    """Average validation loss. eval() + no_grad() disable dropout/BN updates
    and gradient tracking."""
    model.eval()
    running_vloss = 0.
    with torch.no_grad():
        for i, (vinputs, vlabels) in enumerate(validation_loader):
            running_vloss += loss_fn(model(vinputs), vlabels).item()
    return running_vloss / (i + 1)
