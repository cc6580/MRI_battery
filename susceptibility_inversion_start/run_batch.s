#!/bin/bash

# requesting resources and other parameters
# submit job with 'sbatch run_batch.s'
# monitor running jobs using squeue, cancel with scancel, etc

#SBATCH --job-name=inversion_example
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=20:00:00
#SBATCH --mail-type=END
#SBATCH --output=output.txt
##SBATCH --error=error.txt

#PRINCE PRINCE_GPU_COMPUTE_MODE=DEFAULT

# loading relevant environments and packages
module purge
module load anaconda3/5.3.1
source $ANACONDA3_ROOT/etc/profile.d/conda.sh
conda activate /scratch/cc6580/share/inversion_example/inversion_env
#export DISPLAY=""

# for working without X-winndow connection in batch mode
# then do not need agg code in script (convenient)
export MPLBACKEND="agg"

conda info --envs
# conda list

# run your python script
python inversion_example.py
