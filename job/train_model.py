"""
Preemptible training example: FashionMNIST classifier that checkpoints every
epoch and resumes from the latest checkpoint after a preemption / requeue.

Key ideas demonstrated:
  * Save EVERYTHING needed to resume (model, optimizer, epoch), not just weights
  * Write checkpoints atomically (tmp file + rename) so a kill mid-save can't
    leave a corrupt "latest" checkpoint
  * Keep checkpoints per Slurm job ID (the ID survives a requeue)
  * Catch SIGTERM (sent by Slurm on preemption, scontrol requeue, or scancel),
    save, and exit cleanly before the follow-up SIGKILL

Configuration via environment variables (all optional):
  CKPT_DIR    where checkpoints go   (default: ../models/$SLURM_JOB_ID)
  DATA_DIR    where the dataset is   (default: ./data)
  RUNS_DIR    TensorBoard log dir    (default: ../runs/$SLURM_JOB_ID)
  EPOCHS      total epochs to train  (default: 50)
  BATCH_SIZE  training batch size    (default: 32)
  CKPT_EVERY  checkpoint every N epochs (default: 1)

Run `python3 train_model.py --download-only` on a login node to fetch the data
before submitting.
"""
import glob
import os
import re
import signal
import sys
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils import GarmentClassifier, train_one_epoch, evaluate

##############################
# Configuration
##############################
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
RESTART_COUNT = int(os.environ.get("SLURM_RESTART_COUNT", "0"))

CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join("..", "models", JOB_ID))
DATA_DIR = os.environ.get("DATA_DIR", "./data")
RUNS_DIR = os.environ.get("RUNS_DIR", os.path.join("..", "runs", JOB_ID))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "1"))

CKPT_PATTERN = re.compile(r"model_(\d+)\.pt$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


##############################
# SIGTERM handling
##############################
# Slurm sends SIGTERM when a job is preempted, requeued, or cancelled, then
# SIGKILL a short time later. We only set a flag here; the training loop checks
# it after every batch, saves a checkpoint, and exits. Doing the save inside
# the handler itself is risky because the handler can fire mid-optimizer-step.
STOP_REQUESTED = False


def handle_sigterm(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(f"Caught signal {signal.Signals(signum).name}: will save a checkpoint "
        "after the current batch and exit.")


signal.signal(signal.SIGTERM, handle_sigterm)


def stop_requested():
    return STOP_REQUESTED


##############################
# Checkpoint helpers
##############################
def save_checkpoint(model, optimizer, next_epoch, epoch_complete):
    """Atomically save training state.

    next_epoch: the epoch index training should START from on resume.
    epoch_complete: False if this is an emergency save from the middle of an
                    epoch (that epoch will be re-run on resume).
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"model_{next_epoch}.pt")
    tmp_path = path + ".tmp"
    state = {
        "next_epoch": next_epoch,
        "epoch_complete": epoch_complete,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX: readers see old or new, never half
    log(f"Checkpoint saved: {path}")
    return path


def find_checkpoints():
    """Return checkpoint paths sorted newest (highest epoch) first."""
    found = []
    for f in glob.glob(os.path.join(CKPT_DIR, "model_*.pt")):
        m = CKPT_PATTERN.search(f)
        if m:
            found.append((int(m.group(1)), f))
    return [f for _, f in sorted(found, reverse=True)]


def load_latest_checkpoint(model, optimizer):
    """Load the newest readable checkpoint. Returns the epoch to start from."""
    for path in find_checkpoints():
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
        except Exception as e:  # corrupt or incompatible file: try an older one
            log(f"WARNING: could not load {path} ({e}); trying an older checkpoint")
            continue
        note = "" if state["epoch_complete"] else " (mid-epoch save; re-running that epoch)"
        log(f"Loaded {path}{note}")
        return state["next_epoch"]
    return 0


##############################
# Data
##############################
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def get_datasets():
    train = torchvision.datasets.FashionMNIST(DATA_DIR, train=True, transform=transform, download=True)
    val = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, transform=transform, download=True)
    return train, val


def main():
    if "--download-only" in sys.argv:
        get_datasets()
        log(f"Dataset downloaded to {DATA_DIR}")
        return

    # Use the CPUs Slurm gave us (defaults to 1 thread otherwise on some setups)
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    torch.set_num_threads(n_cpus)

    training_set, validation_set = get_datasets()
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)
    log(f"Training set: {len(training_set)} images; validation set: {len(validation_set)} images")

    ##############################
    # Model + optimizer, resuming if a checkpoint exists
    ##############################
    model = GarmentClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start_epoch = load_latest_checkpoint(model, optimizer)

    log("=" * 60)
    if start_epoch > 0:
        log(f"RESUMED from checkpoint at epoch {start_epoch}  "
            f"(job {JOB_ID}, restart #{RESTART_COUNT})")
    else:
        log(f"STARTING FRESH  (job {JOB_ID}, restart #{RESTART_COUNT})")
    log(f"Checkpoints: {CKPT_DIR}   Epochs: {EPOCHS}   Threads: {n_cpus}")
    log("=" * 60)

    if start_epoch >= EPOCHS:
        log("Training already complete. Nothing to do.")
        return

    writer = SummaryWriter(RUNS_DIR)  # same dir across restarts -> one continuous curve

    ##############################
    # Train
    ##############################
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        log(f"EPOCH {epoch + 1}/{EPOCHS}")

        model.train(True)
        avg_loss = train_one_epoch(model, loss_fn, epoch, writer, training_loader,
                                   optimizer, stop_requested=stop_requested)

        if STOP_REQUESTED:
            # Emergency save: the epoch didn't finish, so resume will re-run it
            # starting from these (partially updated) weights.
            save_checkpoint(model, optimizer, next_epoch=epoch, epoch_complete=False)
            writer.close()
            log("Exiting after SIGTERM. Slurm will requeue the job.")
            sys.exit(0)

        avg_vloss = evaluate(model, loss_fn, validation_loader)
        log(f"LOSS train {avg_loss:.4f} valid {avg_vloss:.4f}  ({time.time() - t0:.0f}s)")
        writer.add_scalars("Training vs. Validation Loss",
                           {"Training": avg_loss, "Validation": avg_vloss}, epoch + 1)
        writer.flush()

        # Checkpoint at the end of the epoch. Named by the NEXT epoch to run,
        # so model_5.pt means "5 epochs done, start at epoch index 5".
        # (Also save if SIGTERM arrived during validation.)
        if (epoch + 1) % CKPT_EVERY == 0 or epoch + 1 == EPOCHS or STOP_REQUESTED:
            save_checkpoint(model, optimizer, next_epoch=epoch + 1, epoch_complete=True)

        if STOP_REQUESTED:
            writer.close()
            log("Exiting after SIGTERM. Slurm will requeue the job.")
            sys.exit(0)

    writer.close()
    log("Training complete!")


if __name__ == "__main__":
    main()
