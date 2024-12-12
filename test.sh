#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 4           # number of cores
#SBATCH -t 0-01:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"

module load mamba/latest

source activate TEPCAM

cd ~/CSE559/TEPCAM/

python ./scripts/test.py --file_path="./Data/tcr_split/test.csv" --model_path="./model/TEPCAM_tcr_1.pt" --output_file="./output_tcr_1.csv" --metric_file="./metric_file_tcr_1.csv"

python ./scripts/test.py --file_path="./Data/tcr_split/test.csv" --model_path="./model/TEPCAM_tcr_2.pt" --output_file="./output_tcr_2.csv" --metric_file="./metric_file_tcr_2.csv"

python ./scripts/test.py --file_path="./Data/tcr_split/test.csv" --model_path="./model/TEPCAM_tcr_3.pt" --output_file="./output_tcr_3.csv" --metric_file="./metric_file_tcr_3.csv"

python ./scripts/test.py --file_path="./Data/epi_split/test.csv" --model_path="./model/TEPCAM_epi_1.pt" --output_file="./output_epi_1.csv" --metric_file="./metric_file_epi_1.csv"

python ./scripts/test.py --file_path="./Data/epi_split/test.csv" --model_path="./model/TEPCAM_epi_2.pt" --output_file="./output_epi_2.csv" --metric_file="./metric_file_epi_2.csv"

python ./scripts/test.py --file_path="./Data/epi_split/test.csv" --model_path="./model/TEPCAM_epi_3.pt" --output_file="./output_epi_3.csv" --metric_file="./metric_file_epi_3.csv"