# Pre-Empting and Checkpointing


## What is "pre-empting"?

A preemptible resource is one the system can take back from you at any time, often with only seconds of warning or none. In exchange, those resources are cheaper or easier to get.

Pre-empted resources can be found on:
* The `owners` queue of Sherlock
* Marlowe Basic
* Spot instances in Google Cloud


## Why do we care?

If your code can handle being preempted, you get access to far more resources, and you get them sooner.

## What happens when a Slurm job is preempted?

1. Slurm sends your job **SIGTERM**.
2. After a short grace period (check your partition's `GraceTime`; it may be very short), Slurm sends **SIGKILL**, which can't be caught.
3. With `#SBATCH --requeue`, the job goes back in the queue **with the same job ID**, and `SLURM_RESTART_COUNT` goes up by one.

`scontrol requeue <job_id>` triggers the same sequence on your own job, which is how we test it.

## Robust and Restartable Code

* **State management**
  * Your code should be able to detect which stages to run and which to skip
  * For example: if a file exists, skip the step that creates it
* **Idempotency**
  * Your code should produce the same result no matter how many times you run it
* **Checkpointing**
  * Save *everything* needed to resume: model weights, **optimizer state**, the epoch/step counter, and RNG state, not just the weights
  * Write checkpoints **atomically**: write to `ckpt.tmp`, then rename it to `ckpt.pt`. A kill during the save then leaves the previous checkpoint intact instead of a corrupt one.
  * Put checkpoints on **shared storage** (`$SCRATCH`, `$GROUP_HOME`, Oak), never node-local `/tmp`. A requeued job may land on a different node.
  * Checkpoint **often enough** that a hard kill with no warning costs an amount of work you can live with
* **Graceful shutdown**
  * Catch SIGTERM, finish the current step, save, and exit
  * Keep the handler tiny (just set a flag) and do the save from your main loop
