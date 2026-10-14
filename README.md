# Pre-Emptible Jobs and Checkpointing
Brought to you by the [SDSS Center for Computation](https://sdss-compute.stanford.edu)

In today's workshop we are going to cover:

* Checkpointing in Pytorch
* Pre-Emptible jobs in `owners`
* How to restart gracefully

### Today's Tutorial Agenda:
1. Learn about Checkpointing, Pre-Emptibility, and Restartable Code 
2. Login to Sherlock
3. Submit a pre-emptible job to the `owners` queue
4. Review the job and monitor checkpoints
5. Pre-empt your own job and watch it requeue

Background reading is in [about_checkpoints.md](about_checkpoints.md), and the step-by-step lab is in [job/README.md](job/README.md).

### Repo layout
```
about_checkpoints.md        concepts: preemption, checkpointing, restartable code
env/pytorch.def             Apptainer recipe for the container
job/preemptible_job.submit  Slurm batch script (owners partition, --requeue)
job/train_model.py          training loop with checkpoint/resume + SIGTERM handling
job/utils.py                model, train/eval helpers
job/out/                    Slurm logs land here
```

### Do you have any questions? 
Please reach out to us on our slack channel, `#sdss-compute-users`, at [sdss-compute@stanford.edu](mailto:sdss-compute@stanford.edu).


