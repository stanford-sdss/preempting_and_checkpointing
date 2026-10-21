# Preemptible job tutorial

## Step 1: Log in to a Sherlock shell
Either via [OnDemand](https://ondemand.sherlock.stanford.edu/pun/sys/dashboard)

Or via ssh: `ssh <sunet>@login.sherlock.stanford.edu`

## Step 2: Clone this repository

```bash
git clone https://github.com/MATRICS-Bootcamp/preempting-and-checkpointing.git
cd preempting-and-checkpointing
```

## Step 3: Get the container

Cpy the pre-pulled container from Oak:
```bash
cp /oak/stanford/datasets/sdss/containers/preempting-and-checkpointing/pytorch_latest.sif  ./env/
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
sbatch preemptible_job.submit
```

Write down the job ID that `sbatch` prints. Below it's `<job_id>`.

## Step 6: Watch the job

Use three terminals (or tmux panes):

```bash
# 1. Job state
watch -n 10 squeue --me

# 2. Training log: look for "STARTING FRESH" and "Checkpoint saved"
tail -f out/preempt_example_<job_id>.out

# 3. Checkpoints appearing (one per epoch)
watch -n 10 ls -lt ../models/<job_id>/
```

Each checkpoint is named for the **next** epoch to run: `model_5.pt` means "5 epochs done, resume at epoch 6."

While it runs, look at `train_model.py`. What goes into a checkpoint? Why is it written to a `.tmp` file first?

## Step 7: Preempt the job

Once a few checkpoints exist:

```bash
scontrol requeue <job_id>
```

This does the same thing to your job as a real preemption on `owners`: Slurm sends **SIGTERM**, waits a short grace period, then sends SIGKILL and puts the job back in the queue.

## Step 8: Observe again!

* **Log:** you should see
  ```
  Caught signal SIGTERM: will save a checkpoint after the current batch and exit.
  Checkpoint saved: ../models/<job_id>/model_N.pt
  Exiting after SIGTERM. Slurm will requeue the job.
  ```
  When the job starts again, a new header appears in the **same** log file:
  ```
  RESUMED from checkpoint at epoch N  (job <job_id>, restart #1)
  ```
* **squeue:** the job goes `R` → `PD` → `R`. A requeued job waits in `PD` for a bit (often a minute or two) before it can start again. That's normal.
* **sacct:** every run of the job shows up as its own row:
  ```bash
  sacct -j <job_id> --duplicates --format=JobID,State,Start,End,Elapsed,NodeList
  ```

Try requeueing it a second time. Does the restart count go up?

## Step 9 (optional): TensorBoard

The loss curve continues across restarts because every run writes to `../runs/<job_id>`.

## Step 10: Clean up

```bash
scancel <job_id>
rm -rf ../models/<job_id> ../runs/<job_id> out/preempt_example_<job_id>.*
```

## Things to try

* **Turn off the SIGTERM handler.** Comment out `signal.signal(signal.SIGTERM, handle_sigterm)` in `train_model.py` and requeue mid-epoch. How much work is lost now?
* **Checkpoint less often.** Set `export CKPT_EVERY=5` in the submit script. How much do you lose on a hard kill?
* **Submit a second job.** It starts fresh. Why? (Hint: look at `CKPT_DIR`.)
