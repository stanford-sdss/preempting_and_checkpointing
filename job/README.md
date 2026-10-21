# Preemptible job tutorial

## Step 1: Log in to a Sherlock shell
Either via [OnDemand](https://ondemand.sherlock.stanford.edu/pun/sys/dashboard)

Or via ssh: `ssh <sunet>@login.sherlock.stanford.edu`

## Step 2: Clone this repository

```bash
git clone https://github.com/stanford-sdss/preempting_and_checkpointing.git
cd preempting_and_checkpointing
```

## Step 3: Get the container

From the top of the repo:

```bash
apptainer pull env/pytorch_latest.sif oras://ghcr.io/matrics-bootcamp/pytorch:latest
```

If that takes too long, copy the pre-pulled copy from Oak instead:
```bash
cp /oak/stanford/schools/ees/containers/pytorch_latest.sif ./env/
```

## Step 4: Download the dataset (on the login node)

Fetch FashionMNIST once here, so every job doesn't download it from a compute node:

```bash
cd job
apptainer exec ../env/pytorch_latest.sif python3 train_model.py --download-only
```

## Step 5: Submit the job

Still in the `job/` directory:

```bash
mkdir -p out          # Slurm will NOT create the log directory for you
JOBID=$(sbatch --parsable preemptible_job.submit)
echo $JOBID
```

`--parsable` makes `sbatch` print just the job ID, so we can save it in `$JOBID` and use it in every command below.

> **Lost `$JOBID`?** (New terminal, dropped connection.) Get it back with:
> ```bash
> JOBID=$(squeue --me -h -n preempt_example -o %i | head -1)
> ```

## Step 6: Watch the job

One command opens a tmux session with everything on one screen:

```bash
tmux new-session -s lab "tail -F out/preempt_example_$JOBID.out" \; \
  split-window -h "watch -n 10 squeue --me" \; \
  split-window -v "watch -n 10 ls -lt ../models/$JOBID" \; \
  split-window -v "JOBID=$JOBID bash" \; \
  select-layout main-vertical \; \
  set-option mouse on
```

```
+----------------------------+----------------------+
|                            | squeue: job state    |
|  Training log              +----------------------+
|  (look for STARTING FRESH  | checkpoints in       |
|   and Checkpoint saved)    | ../models/$JOBID     |
|                            +----------------------+
|                            | your shell: run the  |
|                            | next steps here      |
+----------------------------+----------------------+
```

* **Switch panes:** click one (mouse mode is on), or press `Ctrl-b` and then an arrow key.
* **Scroll the log:** scroll with the mouse wheel over it; press `q` to go back to following it.
* **Leave and come back later:** `Ctrl-b` then `d` detaches; `tmux attach -t lab` reattaches. Sherlock has several login nodes, so if you reconnect and the session is missing, ssh to the node you were on (e.g. `ssh sh03-ln01`).

Each checkpoint is named for the **next** epoch to run: `model_5.pt` means "5 epochs done, resume at epoch 6."

While it runs, look at `train_model.py`. What goes into a checkpoint? Why is it written to a `.tmp` file first?

## Step 7: Preempt the job

Once a few checkpoints exist, run this in the shell pane (bottom right):

```bash
scontrol requeue $JOBID
```

This does the same thing to your job as a real preemption on `owners`: Slurm sends **SIGTERM**, waits a short grace period, then sends SIGKILL and puts the job back in the queue.

## Step 8: Observe again!

* **Log:** you should see something like this (your job ID instead of `12345678`):
  ```
  Caught signal SIGTERM: will save a checkpoint after the current batch and exit.
  Checkpoint saved: ../models/12345678/model_4.pt
  Exiting after SIGTERM. Slurm will requeue the job.
  ```
  When the job starts again, a new header appears in the **same** log file:
  ```
  RESUMED from checkpoint at epoch 4  (job 12345678, restart #1)
  ```
* **squeue:** the job goes `R` → `PD` → `R`. A requeued job waits in `PD` for a bit (often a minute or two) before it can start again. That's normal.
* **sacct:** every run of the job shows up as its own row:
  ```bash
  sacct -j $JOBID --duplicates --format=JobID,State,Start,End,Elapsed,NodeList
  ```

Try requeueing it a second time. Does the restart count go up?

## Step 9 (optional): TensorBoard

The loss curve continues across restarts because every run writes to `../runs/$JOBID`.

## Step 10: Clean up

```bash
scancel $JOBID
rm -rf ../models/$JOBID ../runs/$JOBID out/preempt_example_$JOBID.*
tmux kill-session -t lab
```

## Things to try

* **Turn off the SIGTERM handler.** Comment out `signal.signal(signal.SIGTERM, handle_sigterm)` in `train_model.py` and requeue mid-epoch. How much work is lost now?
* **Checkpoint less often.** Set `export CKPT_EVERY=5` in the submit script. How much do you lose on a hard kill?
* **Submit a second job.** It starts fresh. Why? (Hint: look at `CKPT_DIR`.)
