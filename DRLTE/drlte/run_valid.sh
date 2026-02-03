#!/bin/bash -x 
seed=$1
mini_batch=$2
len_circle=$3
tm_circle=$4
explore_epochs=$5  
epsilon_steps=$6
explore_decay=$7
ckpt_idx=$8
topoName=$9
logFile=${10}

# debug
echo "seed is ${seed}, mini_batch is ${mini_batch}, len_circle is ${len_circle}, tm_circle is ${tm_circle}, explore_epochs is ${explore_epochs}, epsilon_steps is ${epsilon_steps}, explore_decay is ${explore_decay}, ckpt_idx is ${ckpt_idx}, topoName is ${topoName}, logFile is ${logFile} "

# seed=3
# mini_batch=16
# tm_circle=5
# len_circle=100
# explore_epochs=0
# epsilon_steps=9437
# explore_decay=0
# ckpt_idx=1
# topoName=Abi

# lpPerformFile=../../inputs/Perf_${topoName}_R5H_V1_S.txt
# file_name=${topoName}_R5H_V1

lpPerformFile=../inputs/${topoName}_pf_valid500.txt
file_name=${topoName}_valid500

########for test#########
#epochs_test=1
#episodes_test=500

test_epochs=1
test_episodes=500

CUDA_VISIBLE_DEVICES='-1' \
python sim-ddpg_sol10.py \
--lpPerformFile=${lpPerformFile} \
--stamp_type=../../${logFile}/validRes/${topoName}-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}-${ckpt_idx} \
--ckpt_path=../${logFile}/ckpoint/${topoName}-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}/ckpt-${ckpt_idx}0000 \
--random_seed=${seed} --mini_batch=${mini_batch} --epochs=${test_epochs} --len_circle=1 \
--tm_circle=1 \
--explore_epochs=${explore_epochs} \
--epsilon_steps=${epsilon_steps} \
--explore_decay=${explore_decay} \
--episodes=${test_episodes} \
--file_name=${file_name} \
--offline_flag=True \
--epsilon_begin=0. \
--is_train=False   # modified
