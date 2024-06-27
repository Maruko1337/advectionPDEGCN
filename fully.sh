#!/bin/bash
#SBATCH --job-name="fully"
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x_%j.out
echo "Cuda device: $CUDA_VISIBLE_DEVICES"
echo "======= Start memory test ======="
nvidia-smi

module load anaconda3/2022.10

MY_CONDA_ENV="gpua100"

## Check if MY_CONDA_ENV has already been  created
CHK_ENV=$(conda  env list | grep $MY_CONDA_ENV | awk '{print $1}')

echo "CHK_ENV: $CHK_ENV"
if [ "$CHK_ENV" =  "" ]; then
        ## if MY_CONDA_ENV does not exist
        echo "$MY_CONDA_ENV doesn't exist, create it..."
        conda create --yes  --name $MY_CONDA_ENV -c anaconda  numpy pandas tensorflow-gpu
        conda activate $MY_CONDA_ENV
else
        ## if MY_CONDA_ENV already exist
        echo "MY_CONDA_ENV exists, activate $MY_CONDA_ENV"
        #conda init bash
        conda activate $MY_CONDA_ENV
fi

echo "test "
which python
python ./src/pdegcn_fully.py