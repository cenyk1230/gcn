#!/bin/bash

# input arguments
DATA="${1-MUTAG}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
fold=${2-1}  # which fold as testing data
seed=${3-1}  # random seed
gpu=${4-3}   # gpu number

# general settings
slice_k=0.6  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
bsize=50  # batch size
use_deg=0

# dataset-specific settings
case ${DATA} in
MUTAG)
  bsize=20
  use_deg=1
  num_epochs=100
  learning_rate=0.0001
  ;;
NCI1)
  bsize=100
  num_epochs=200
  learning_rate=0.0001
  ;;
DD)
  bsize=50
  num_epochs=200
  learning_rate=0.0001
  ;;
PTC)
  slice_k=0.6
  bsize=50
  use_deg=1
  num_epochs=50
  learning_rate=0.0001
  ;;
PROTEINS)
  bsize=50
  num_epochs=100
  learning_rate=0.0001
  ;;
COLLAB)
  use_deg=1
  num_epochs=100
  learning_rate=0.0001
  slice_k=0.9
  ;;
IMDBBINARY)
  num_epochs=300
  learning_rate=0.0001
  slice_k=0.9
  ;;
IMDBMULTI)
  num_epochs=100
  learning_rate=0.0001
  slice_k=0.9
  ;;
esac

result_file="result_${seed}.txt"
if [ ${fold} == 0 ]; then
  rm $result_file
  echo "Running 10-fold cross validation"
  start=`date +%s`
  for i in $(seq 1 10)
  do
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        -seed $seed \
        -data $DATA \
        -fold $i \
        -use_deg $use_deg \
        -learning_rate $learning_rate \
        -num_epochs $num_epochs \
        -slice_k $slice_k \
        -batch_size $bsize \
        -gpu $gpu
  done
  stop=`date +%s`
  echo "End of cross-validation"
  echo "The total running time is $[stop - start] seconds."
  echo "The accuracy results for ${DATA} are as follows:"
  cat $result_file
  echo "Average accuracy is"
  cat $result_file | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
else
  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
      -seed $seed \
      -data $DATA \
      -fold $fold \
      -use_deg $use_deg \
      -learning_rate $learning_rate \
      -num_epochs $num_epochs \
      -slice_k $slice_k \
      -batch_size $bsize \
      -gpu $gpu
fi
