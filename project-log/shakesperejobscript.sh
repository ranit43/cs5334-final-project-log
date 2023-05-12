#!/bin/bash
#SBATCH -o shakespere-13824.out%j
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH -t 0:45:00
#SBATCH --gpus=v100-32:8
#SBATCH --mail-user=rdakash@miners.utep.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

#type 'man sbatch' for more information and options
#this job will ask for 1 full v100-32 GPU node(8 V100 GPUs) for 5 hours
#this job would potentially charge 40 GPU SUs

#echo commands to stdout
set -x

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

cd /ocean/projects/cis230018p/rakash/projects/nano-gpt-git-v0/nanoGPT/
export HF_DATASETS_CACHE="/ocean/projects/cis230018p/rakash/"

#run pre-compiled program which is already in your project space
module load cuda
module load anaconda3
conda activate /ocean/projects/cis230018p/rakash/rda-ngpt-env-v1

python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char