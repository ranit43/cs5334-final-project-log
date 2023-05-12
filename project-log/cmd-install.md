```
pip install transformers

pip install datasets
```

pip install tiktoken

pip install wandb

Home Directory:
/jet/home/rakash

project directory

sbatch shakesperejobscript.sh
-p GPU --gpus=v100-32:8 shakesperejobscript.sh

-o slurm-jobid.out

--mail-type=type
--mail-user=username

shakesperejobscript.sh

cd /ocean/projects/cis230018p/rakash/projects/nano-gpt-git-v0/nanoGPT/

cd \_\_ nanoGPT

interact -p GPU-shared --gres=gpu:v100-32:2 -t 1:00:00

module load anaconda3

module load cuda

conda activate /ocean/projects/cis230018p/rakash/rda-ngpt-env-v1

python data/shakespeare_char/prepare.py

python train.py config/train_shakespeare_char.py

python sample.py --out_dir=out-shakespeare-char
