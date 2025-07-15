#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:a40:2
#SBATCH --time=2880
#SBATCH --cpus-per-task=64
#SBATCH --job-name=lfq
#SBATCH --output=%j_output.txt
#SBATCH --error=%j_error.txt

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source .venv/bin/activate

# Generate a random master port to avoid collision
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Using MASTER_PORT="$MASTER_PORT

code_length=256
vocab_size=14
patch_size=8

torchrun --nproc-per-node=2 --master_port=$MASTER_PORT -m flowmo.train \
    --experiment-name "flowmo_lfq_p${patch_size}_l${code_length}_v$(( 2 ** vocab_size ))_pretrain" \
    model.context_dim=${vocab_size} model.codebook_size_for_entropy=$(( vocab_size )) model.quantization_type=lfq \
    model.code_length=${code_length} \
    model.patch_size=${patch_size} \
    model.mup_width=4 \
    data.batch_size=16 \
    data.imagenet_data_source=wds \
    trainer.max_steps=400000 \
    trainer.checkpoint_every=10000 \
    trainer.keep_every=10000

echo "Done"
exit 0
