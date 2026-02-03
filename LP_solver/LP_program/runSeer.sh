### Abi
python3 seerSolver.py \
--graphFile=../originInput/Abi/Abi.txt \
--tmFile=../originInput/Abi/Abi_OR_tm500.txt \
--perfFile=../LP_output/Abi/Util_Abi_seer_500.txt \
--scaleCapac=50  #the scaling factor of link bandwidth

### GEA
python3 seerSolver.py \
--graphFile=../originInput/GEA/GEA.txt \
--tmFile=../originInput/GEA/GEA_OR_tm500.txt \
--perfFile=../LP_output/GEA/Util_GEA_seer_500.txt \
--scaleCapac=625

