#! /bin/bash
#SBATCH --job-name=cluster
#SBATCH -o job_out.%j.out
#SBATCH -e job_error.%j.out
#SBATCH --partition grande
#SBATCH -n 96
#SBATCH --time 3-00:00:00

hostname
source ~/.bashrc
conda activate pyswcloader

python test_cluster.py --data_path "/home/dongzhou/data/mouse/PFC_20240527" --save_path "/home/dongzhou/data/mouse/scores/PFC_20240527_cluster_result" --workers 96 --n_cluster 64