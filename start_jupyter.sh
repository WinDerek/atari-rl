#!/bin/bash

BEGIN=$(date +"%Y%m%d_%H%M%S")
mkdir -p ./logs/jupyter/${BEGIN}
export PASSWORD=jupyter
setsid stdbuf -i0 -o0 -e0 /home/derek/miniconda3/envs/atari_env/bin/jupyter-lab --config jupyter_config.py --ip 0.0.0.0 --port 41589 --allow-root > ./logs/jupyter/${BEGIN}/jupyter.log 2>&1 &
PROCESS_ID=$!
echo PID: ${PROCESS_ID}
echo ${PROCESS_ID} > ./logs/jupyter/${BEGIN}/jupyter_pid.txt
tail -f ./logs/jupyter/${BEGIN}/jupyter.log
