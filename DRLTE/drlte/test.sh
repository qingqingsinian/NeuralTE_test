#!/bin/bash -x
modelHyperFile=$1   # /home/hesy/projects/mars/DRLTE/log/validRes/Cernet2/modelIndex.txt
# /home/hesy/pro/mars/DRLTE/log_trueTM/validRes/GEA-trueTM/modelIndex.txt
logFile=log # log_trueTM

while read line 
do
	seed=$(echo $line|awk '{print $1}');
	mini_batch=$(echo $line|awk '{print $2}');
	len_circle=$(echo $line|awk '{print $3}');
	tm_circle=`echo $line|awk '{print $4}'`
	explore_epochs=$(echo $line|awk '{print $5}');  
	epsilon_steps=$(echo $line|awk '{print $6}');
	explore_decay=$(echo $line|awk '{print $7}');
	ckpt_idx=$(echo $line|awk '{print $8}');
	topoName=$(echo $line|awk '{print $9}');
	# dim=$(echo $line|awk '{print $10}');

    ############  debug  ############	

	# lpPerformFile=../inputs/pr_${topoName}_seer_${dim}_valid500.txt
	# file_name=${topoName}_save${dim}_valid500
	lpPerformFile=../inputs/${topoName}_pf_valid500.txt
	# file_name=${topoName}_trueTM_test500_drlte
	# file_name=${topoName}_trueTM_test500
	file_name=${topoName}_valid500

	test_epochs=1
	test_episodes=500

	CUDA_VISIBLE_DEVICES='-1' \
	python sim-ddpg_sol10.py \
	--lpPerformFile=${lpPerformFile} \
	--stamp_type=../../${logFile}/test/${topoName}-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}-${ckpt_idx} \
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
	--is_train=False \
	> ../${logFile}/validRes-monitorLogs/${topoName}/test-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}-${ckpt_idx}.log  2>&1

done < ${modelHyperFile}

	# --stamp_type=../../${logFile}/test/${topoName}-${dim}-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}-${ckpt_idx} \
    # --ckpt_path=../${logFile}/ckpoint/${topoName}-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}/ckpt-${ckpt_idx}0000 \

# ! fei's version needs change here
# from 
    # --ckpt_path=../${logFile}/ckpoint/Abi-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}/ckpt-${ckpt_idx}0000 \
# to
	# --ckpt_path=../${logFile}/ckpoint/${topoName}-${dim}-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}/ckpt-${ckpt_idx}0000 \


<<Comment # drl-te版本
CUDA_VISIBLE_DEVICES='-1' \
python sim-ddpg.py \
--lpPerformFile=${lpPerformFile} \
--stamp_type=../../${logFile}/test/drl-te \
--ckpt_path=../${logFile}/ckpoint/drl-te/ckpt \
--random_seed=${seed} --mini_batch=${mini_batch} --epochs=${test_epochs} --len_circle=1 \
--tm_circle=1 \
--explore_epochs=${explore_epochs} \
--epsilon_steps=${epsilon_steps} \
--explore_decay=${explore_decay} \
--episodes=${test_episodes} \
--file_name=${file_name} \
--offline_flag=True \
--epsilon_begin=0. \
--is_train=False
# --is_train=False \
# > ../${logFile}/validRes-monitorLogs/${topoName}/test-${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}-${ckpt_idx}.log  2>&1
Comment