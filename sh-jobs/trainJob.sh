#!/bin/bash
#SBATCH --job-name=flower_train
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=train_%j.log
#SBATCH --error=ddl_%j.err

source /opt/miniconda3.10/bin/activate

# Aktivirajte okruženje
source /home/ivanam/.conda/envs/flwr-client/bin/activate

# Instalirajte Flower (ako već nije instaliran)
pip install flwr

# Pokrenite Python klijent
python client1.py

# Deaktivirajte okruženje nakon završetka
conda deactivate
