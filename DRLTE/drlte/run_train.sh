#!/bin/bash -x
# lpPerformFile=../inputs/GEA_restricted_mcf_SHR_3-true-tm-train-2000.txt
# lpPerformFile=../inputs/GEA_restricted_mcf_SHR_3-true-tm-test-500.txt

# lpPerformFile=../inputs/Abilene/pf_trueTM_test.txt
# file_name=Abilene/Abi_trueTM500

# debug=0
# if [ debug==1 ]; then
#     lpPerformFile=../inputs/pf_trueTM_train4000.txt
#     file_name=Abi_trueTM_train4000
#     seed=3
#     mini_batch=32
#     tm_circle=200
#     len_circle=2
#     explore_epochs=0 
#     epsilon_steps=9437
#     explore_decay=0
#     topoName=Abi
#     echo "**************training debug**************"
# else
    seed=$1
    mini_batch=$2
    len_circle=$3
    tm_circle=$4
    explore_epochs=$5  
    epsilon_steps=$6
    explore_decay=$7
    topoName=$8
    # ckpath=$9
# fi

# === Abi

if [ $topoName = 'Cernet2' ]; then
    echo processing topo of ${topoName}_3000
    lpPerformFile=../inputs/${topoName}_pf_train3000.txt
    file_name=${topoName}_train3000
else
    echo processing topo of ${topoName}_4000
    lpPerformFile=../inputs/${topoName}_pf_train4000.txt
    file_name=${topoName}_train4000
fi

# echo "Please Check：1. TM input file has been changed 2. ckpt path is given"
echo "请检查：1. 输入的TM文件是否修改 2. 是否给了要加载的ckpt路径"
echo "(seed,mini_batch,len_circle,tm_circle,explore_epochs,epsilon_steps,explore_decay,topoName) is (${seed},${mini_batch},${len_circle},${tm_circle},${explore_epochs},${epsilon_steps},${explore_decay},${topoName})"
echo "(lpPerformFile,file_name) is ${lpPerformFile},${file_name}"


#*train TrueTM
CUDA_VISIBLE_DEVICES='-1' \
python sim-ddpg_sol10.py \
--lpPerformFile=${lpPerformFile} \
--len_circle=${len_circle} \
--tm_circle=${tm_circle} \
--stamp_type=${topoName}-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay} \
--random_seed=${seed} \
--mini_batch=${mini_batch} \
--explore_epochs=${explore_epochs} \
--epsilon_steps=${epsilon_steps} \
--explore_decay=${explore_decay} \
--file_name=${file_name} \
--offline_flag=True \
--is_train=True # > ../log/monitorLogs/collected_GEA/${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}.log  2>&1  &

# --ckpt_path=${ckpath} \
