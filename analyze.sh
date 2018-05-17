#########################################################################
# File Name: analyze.sh
# Author: cenyk1230
# mail: cenyk1230@qq.com
# Created Time: 2018年04月18日 星期三 16时52分18秒
#########################################################################
#!/bin/bash

for i in $(seq 1 10)
do
    result_file="result_${i}.txt"
    cat $result_file | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
done

