#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=ben.coombs21@imperial.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/bac21/steak-generation/ip/environments/cloudEnv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
python3 /vol/bitbucket/bac21/steak-generation/relu_fields/train_sh_based_voxel_grid_with_posed_images.py -d /vol/bitbucket/bac21/steak-generation/relu_fields/data/triangle -o /vol/bitbucket/bac21/steak-generation/relu_fields/logs/rf/triangle/
python3 /vol/bitbucket/bac21/steak-generation/relu_fields/render_sh_based_voxel_grid.py -i /vol/bitbucket/bac21/steak-generation/relu_fields/logs/rf/triangle/saved_models/model_final.pth -o /vol/bitbucket/bac21/steak-generation/relu_fields/output/multiview_cloud/triangle
uptime