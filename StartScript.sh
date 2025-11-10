#!/bin/bash
#SBATCH --partition=intelsr_long
#SBATCH --ntasks=32
#SBATCH --mem=50G
#SBATCH --job-name=MyPyJob
#SBATCH --output=out/run.%A.out
#SBATCH --account=ag_hiskp_funcke

# The modules are loaded in the ~/.bash_profile

echo -e "Start $(date +"%F %T") | $SLURM_JOB_ID $SLURM_JOB_NAME | $(hostname) | $(pwd) \n"

echo $SLURM_JOB_ID >"out/latest_id"

# shellcheck disable=SC1090
source ~/PycharmProjects/QC-MasterThesis/.venv_intel_3_12_3/bin/activate

echo "Used python interpreter:"
which python
echo "Python version:"
python --version
module list
echo "Start of Program:"
echo

#LOG_INTERVAL=30
#nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l $LOG_INTERVAL > out/gpu_usage_$SLURM_JOB_ID.log &
#GPU_LOG_PID=$!

python main.py --id $SLURM_JOB_ID

#kill $GPU_LOG_PID

echo -e "End $(date +"%F %T") | $SLURM_JOB_ID $SLURM_JOB_NAME | $(hostname) | $(pwd) \n"