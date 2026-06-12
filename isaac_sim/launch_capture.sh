#!/bin/bash
rm -f /dev/shm/sem.carbonite-sharedmemory 2>/dev/null
export CUDALIB="/home/andyee/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle"
export LD_LIBRARY_PATH="$CUDALIB/nvidia/nvjitlink/lib:$CUDALIB/nvidia/cusparse/lib:$CUDALIB/torch/lib:$LD_LIBRARY_PATH"
cd /home/andyee/isaacsim
./isaac-sim.sh --exec /home/andyee/Developer/PG-JY/vision_pose_benchmark/scripts/01_capture_sim_rgbd_dataset.py --scenario desk --num_trials 10 --seed 42
