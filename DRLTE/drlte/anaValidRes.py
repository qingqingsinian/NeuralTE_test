from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
import os
import os.path as osp
import sys
from collections import defaultdict

# modified from g1-validationPerf.ipynb
#! needs change
# parDir="/home/hesy/pro/mars/DRLTE/log_g14/validRes"
# true_exist_num=690  # trueTM
# topoName="Abi"

# hyper
# TODO 参数存在不一样
seeds =[3,66,102]
# seeds =[66] # for fei's version
# mini_batchs =[16,64]    
mini_batchs =[32]  # for fei's version
len_circles =[100,1000]
tm_circles =[5,10]
explore_epochss=[0,1]

def parseAndSave( key ,topoName,expKind,parDir ):
    # 102-64-100-5-0-9437-0-2
    seed,mini_batch,len_circle,tm_circle,explore_epochs,epsilon_steps,explore_decay,ckpt_idx = key.split("-")

    resPathDir = osp.join(parDir,f"{topoName}-{expKind}" )  # ,"modelIndex.txt"
    if not osp.exists(resPathDir):
        os.makedirs(resPathDir)

    print(f"save in to {resPathDir}/modelIndex.txt")
    with open( osp.join(resPathDir,"modelIndex.txt") ,"w") as f:
        print(f"{seed} {mini_batch} {len_circle} {tm_circle} {explore_epochs} {epsilon_steps} {explore_decay} {ckpt_idx} {topoName} {expKind}",file =f)
        """
        print(f"seed {seed}")
        print(f"mini_batch {mini_batch}")
        print(f"len_circle {len_circle}")
        print(f"tm_circle {tm_circle}")
        print(f"explore_epochs {explore_epochs}")
        print(f"epsilon_steps {epsilon_steps}")
        print(f"explore_decay {explore_decay}")
        print(f"ckpt_idx {ckpt_idx}")
        print(f"topoName {topoName}")
        print(f"dim {expKind}")
        """

def main(parDir,true_exist_num,topoName,maxModleNum,expKind):
    ### 组合超参获取文件名
    model_dirPaths=[]   # without model index suffix
    for seed in seeds :
        for mini_batch in mini_batchs :
            for len_circle  in len_circles :
                for tm_circle in tm_circles :
                        for explore_epochs in explore_epochss :
                            # fei's version
                            dirName = f"{topoName}-{expKind}-{seed}-{mini_batch}-{len_circle}-{tm_circle}-{explore_epochs if explore_epochs==0 else len_circle*tm_circle }-{9437 if explore_epochs==0 else 2700}-"
                            # dirName = f"{topoName}-{seed}-{mini_batch}-{len_circle}-{tm_circle}-{explore_epochs if explore_epochs==0 else len_circle*tm_circle }-{9437 if explore_epochs==0 else 2700}-"
                            if explore_epochs:
                                dirName += f"{94 if len_circle*tm_circle==1000 else 189}"  
                            else:
                                dirName += f"{0}"  
                            model_dirPaths.append( osp.join( parDir,dirName)  )

    # 核算一共有多少文件
    existNum = 0
    means_pr,means_util= defaultdict(dict),defaultdict(dict)
    mins_pf = defaultdict(tuple)
    min_pf_all, min_index_all = 10 ,""   # 对于所有模型的所有迭代结果

    for p in model_dirPaths:
        min_pf ,min_index = 10 , ""    # 对于某个模型所有的迭代轮数的结果
        # fei's version
        key = p.split(f'{expKind}')[-1][1:]
        # key = p.split(topoName)[-1][1:]
        for modelIndex in range(1,maxModleNum+1):
            path = p+f"-{modelIndex}"
            if osp.exists( path):
                existNum+=1
                pf_mean, util_mean= np.mean(np.loadtxt( osp.join(path,"perfm.log") )),np.mean(np.loadtxt( osp.join(path,"util.log") ))
                means_pr[key][modelIndex], means_util[key][modelIndex]= pf_mean, util_mean

                if min_pf> pf_mean :
                    min_pf,min_index = pf_mean,f"{key}-{modelIndex}"
        
        if min_pf_all > min_pf:
            min_pf_all,min_index_all = min_pf,min_index
        # mins_pf[key] = (min_index,min_pf) # 记录最小的index以及表现值 # TODO jupyter上可以可视化图像的情况
    assert existNum==true_exist_num, f"wrong existNum of {existNum}"

    parseAndSave( min_index_all, topoName , expKind,parDir)
    print(f"for confirm: min_pf_all is :\n{min_pf_all}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="~")
    parser.add_argument( '--parDir',dest="parDir",help="相应的validRes的绝对路径",type=str , default="/home/hesy/pro/mars/DRLTE/log_time_space_abi/validRes")
    parser.add_argument( '--true_exist_num',dest="true_exist_num",help="存在的所有的模型的数量",type=int ,default=100)    # abi space time 200 196
    parser.add_argument( '--topoName', dest="topoName",help="Abi or GEA",type=str ,default="Abi")
    parser.add_argument( '--maxModleNum', dest="maxModleNum",help="在验证集上参与的模型的序列号的最大值",type=int,default=40 )
    parser.add_argument( '--expKind',dest="expKind",help="本次实验的内容:trueTM,space,time,randTM",type=str,default="Space" )
    args = parser.parse_args()
    print(f"args is {args}")
    main(args.parDir,args.true_exist_num,args.topoName,args.maxModleNum,args.expKind)
