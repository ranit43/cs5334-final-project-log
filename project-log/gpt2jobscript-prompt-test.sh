#!/bin/bash
#SBATCH -o gpt2-af-train-may10-t-01.out%j
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH -t 2:45:00
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

# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_reduced.py

# missing and created error on interact  python train.py eval_gpt2 

# new to take a sample.
# python sample.py 

# cd /ocean/projects/cis230018p/rakash/projects/nano-gpt-git-v0/nanoGPT/data/shakespeare

# python prepare.py

cd /ocean/projects/cis230018p/rakash/projects/nano-gpt-git-v0/nanoGPT/

# python train.py config/finetune_shakespeare_gpt2.py

# python sample.py --out_dir=out-shakespeare

python sample.py \
    --init_from=gpt2 \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100 \
    --out_dir=out-prompt

python sample.py \
    --init_from=gpt2 \
    --start="Please give me a popular recipe?" \
    --num_samples=5 --max_new_tokens=100 \
    --out_dir=out-prompt

python sample.py \
    --init_from=gpt2 \
    --start="How to complement your partner?" \
    --num_samples=5 --max_new_tokens=100 \
    --out_dir=out-prompt

python sample.py \
    --init_from=gpt2 \
    --start="Please write me a story that ends with a happily ever after?" \
    --num_samples=5 --max_new_tokens=100 \
    --out_dir=out-prompt

python sample.py \
    --init_from=gpt2 \
    --start="Please explain how a large language model works on a high level." \
    --num_samples=5 --max_new_tokens=100 \
    --out_dir=out-prompt

python sample.py \
    --init_from=gpt2 \
    --start="Please write a poem about lonely american life?" \
    --num_samples=5 --max_new_tokens=100 \
    --out_dir=out-prompt

python sample.py \
    --init_from=gpt2 \
    --start="Please tell me a interesting joke that includes academia?" \
    --num_samples=5 --max_new_tokens=100 \
    --out_dir=out-prompt


python sample.py \
    --init_from=gpt2 \
    --start="Will AI take over the control of World and human civilization?" \
    --num_samples=5 --max_new_tokens=100 \
    --out_dir=out-prompt

python sample.py \
    --init_from=gpt2 \
    --start="How to be good at Distributed Machine Learning?" \
    --num_samples=5 --max_new_tokens=100 \
    --out_dir=out-prompt
