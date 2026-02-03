#!/bin/bash -x
echo "topoName needs change with varied topologies"
# topoName=Cernet2
topoName=Abi
explore_epochs=0
epsilon_steps=0
explore_decay=0

mkdir -p ../log/monitorLogs/${topoName}

for seed in 102 66 #3 
do
    for mini_batch in 16 32
    do
        for len_circle in 100 1000
        do
            for tm_circle in 5 #10
            do
                for  explore_epochs in 0 1
                do
                    if ((explore_epochs == 0));
                    then
                        epsilon_steps=9437
                        explore_decay=0  #
						#echo ${epochs} ${loop_tm} ${explore_epochs} ${epsilon_steps}
                    else
                        explore_epochs=`expr $len_circle \* $tm_circle`
                        epsilon_steps=2700   #cannot be zero, since that in simddpg.py-pretrain function: self.__beta += (1 - self.__beta) / EP_ST                  
					    if ((`expr $len_circle \* $tm_circle` == 1000));
                        then
                            explore_decay=94
                        else
                            explore_decay=189
                        fi 
                    fi
                    bash -x run_train.sh ${seed} ${mini_batch} ${len_circle} ${tm_circle} ${explore_epochs} ${epsilon_steps} ${explore_decay} ${topoName} \
                    > ../log/monitorLogs/${topoName}/${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}.log  2>&1  &                            
                    # > ../log/monitorLogs/Abi/${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}.log  2>&1  &                            
                    # echo "bash run_train.sh ${seed} ${mini_batch} ${len_circle} ${tm_circle} ${explore_epochs} ${epsilon_steps} ${explore_decay} \
                    # > ../log/monitorLogs/${seed}-${mini_batch}-${len_circle}-${tm_circle}-${explore_epochs}-${epsilon_steps}-${explore_decay}.log"
                done 
            done
        done
    done
done
