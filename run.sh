#!/bin/bash
#SBATCH --job-name=run_python
#SBATCH --output=run_%j.log
#SBATCH --error=run_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

source /home/user18/miniconda3/etc/profile.d/conda.sh
conda activate stllm

cd /home/user18/binhnkt
# python multiscale_runner.py
# python vizuali_scale.py 
# python vizua_scaleB.py
# python vizua_scaleA.py
# python moirai_vizua.py
# python moirai_scaleA2.py
python vizua_A_trend_seasonal.py
# python vizuaAfrequency.py