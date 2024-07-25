#!/bin/bash
#SBATCH --job-name=flower_server
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=server_%j.log
#SBATCH --error=server_%j.err

# Aktivirajte Miniconda okruženje
source /opt/miniconda3.10/bin/activate

# Ako imate specifično okruženje, aktivirajte ga
source /home/ivanam/.conda/envs/ddu2023/bin/activate

# Instalirajte Flower (ako već nije instaliran)
pip install flwr

# Pokrenite Flower server
python server.py

# Opciono: Deaktivirajte virtualno okruženje nakon završetka
conda deactivate
