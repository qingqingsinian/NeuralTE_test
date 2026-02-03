# 环境准备

查看requirements.txt

选择拓扑： GEANT（23，36） 和 Abi(12, 15);
如何选择：
当更换拓扑时，只需要修改训练（train.sh）和推理(valid.sh)脚本的${topoName} 即可。



# 训练

## 批量运行

bash train.sh  （train.sh会循环调用run_train.sh）

1）后台运行

2）运行的log信息存储在../train_abi_log/中

3）训练的中间结果（performance ratio）保存在 ```../log/log/hyper1-hyper2-hyper3..-hyperx``` 文件夹中, 由run_train.sh的--stamp_type参数控制


## 参数介绍

在原有的超参数 ```seed， mini_batch，epochs，loop_tm```，额外引入了三个辅助超参数 explore_epochs，epsilon_steps，explore_decay 来完成具有正常探索功能的 模型训练。 探索分为两种模式：①逐渐衰弱 和 ②周期性探索。


| 超参           | 意义                                                        |
| -------------- | ----------------------------------------------------------- |
| seed           | 随机种子                                                    |
| mini_batch     | mini-batch size                                             |
| epochs         | 和len_circle等价, 一个tm组(包含loop_tm个tm)要循环训练的次数 |
| loop_tm        | 和tm_circle等价，tm组包含的tm的总体数量                     |
| explore_epochs | 控制进入哪种探索模式。                                      |

epsilon_steps和explore_decay都是用来控制探索率的衰弱速度，分别在第一种和第二种模式中起作用，并且值越小，衰弱得越快。
explore_epochs当0时，进入第一种探索模式；大于0时，进入第二种模式。

**探索模式介绍**

目前，在第一种模式下，我们设置50000 steps后，探索率衰退到0.005；

在第二种模式下，我们设置:

1）当epoch(len_circle)和loop_tm(tm_circle)的乘积等于1000时， 在训练500steps后，探索率衰退到0.005(也就是epsilon_steps为2700)； 

2）当epoch和loop_tm的乘积 等于10000时, 在训练500steps后，探索率衰退到0.005(也就是epsilon_steps为9437)。

默认用第一种探索模式。 

控制训练的总的迭代次数: episodes和epoch的乘积
当然，也可以设成很大的值，那么想提前终止程序运行，需要主动kill进程。



# 推理

## 批量运行

bash valid.sh (valid.sh会不断循环run_valid.sh)

除了上述训练中所用到的参数，还会引入一个参数，ckpt_idx来遍历每组参数的所有ckpoint。

test性能结果保存在../DRLTE/log/validRes/， 由run_test.sh的--stamp_type参数控制.

另外，test_epoch=1, test_episode=500 用来控制总的推理test的步数。


# input 文件介绍

输入文件都在DRLTE/inputs/中

* 文件一： 
  \${topoName}\_pf\_trueTM\_train4000.txt: 记录用线性规划求解得到的最优解（最大链路利用率）。 此值被用作计算reward的分母。 topoName指示拓扑名字，存储在当前topoName下，
  该文件需要在运行脚本中指定: lpPerformFile=../inputs/\${topoName}\_pf\_train4000.txt
* 文件二：
  \${topoName}\_train4000，记录候选路径和流量矩阵。 其中topoName指示拓扑名字，存储在当前topoName下，
  该文件也需要在运行脚本中指定:
  file_name=\${topoName}\_train4000

* 文件三： 拓扑文件 
  需要在运行脚本中指定：topoName=GEA

* 其他超参数已写死，默认值如下：

  ```bash
  seed = 66; mini_batch=16; 
  len_circle=1000; tm_circle=5; 
  explore_epochs=0; epsilon_steps=9437; 
  explore_decay=0
  ```

  
