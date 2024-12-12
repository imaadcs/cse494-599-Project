#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 4           # number of cores
#SBATCH -G 4            # number of GPUs 
#SBATCH -t 0-16:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"

module load mamba/latest

source activate TEPCAM

cd ~/CSE559/TEPCAM/

python scripts/train.py --input_file="./Data/tcr_split/train.csv" --model_name="TEPCAM_tcr_1" --epoch=50 --learning_rate=5e-4 --GPU_num=4

python scripts/train.py --input_file="./Data/tcr_split/train.csv" --model_name="TEPCAM_tcr_2" --epoch=50 --learning_rate=1e-4 --GPU_num=4

python scripts/train.py --input_file="./Data/tcr_split/train.csv" --model_name="TEPCAM_tcr_3" --epoch=50 --learning_rate=1e-3 --GPU_num=4

python scripts/train.py --input_file="./Data/epi_split/train.csv" --model_name="TEPCAM_epi_1" --epoch=50 --learning_rate=5e-4 --GPU_num=4

python scripts/train.py --input_file="./Data/epi_split/train.csv" --model_name="TEPCAM_epi_2" --epoch=50 --learning_rate=1e-4 --GPU_num=4

python scripts/train.py --input_file="./Data/epi_split/train.csv" --model_name="TEPCAM_epi_3" --epoch=50 --learning_rate=1e-3 --GPU_num=4