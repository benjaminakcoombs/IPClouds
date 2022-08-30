#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=ben.coombs21@imperial.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/bac21/steak-generation/ip/environments/cloudEnv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
python3 /vol/bitbucket/bac21/steak-generation/ip/pixel-nerf/train/trainMask.py -n fullRenderMask -c /vol/bitbucket/bac21/steak-generation/ip/pixel-nerf/conf/exp/fullRenderMask.conf -D /vol/bitbucket/bac21/steak-generation/fullRender/fullRender/final/fullRender -V 1
uptime