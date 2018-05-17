#########################################################################
# File Name: run_10.sh
# Author: cenyk1230
# mail: cenyk1230@qq.com
# Created Time: 2018年04月18日 星期三 13时31分09秒
#########################################################################
#!/bin/bash

SHELL=$1
DATA=$2

for i in $(seq 1 10)
do
    NAME="nohup_${i}.out"
    sleep 1
    nohup ./${SHELL} ${DATA} 0 ${i} > ${NAME} &
done

