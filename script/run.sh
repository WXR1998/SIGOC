export CUDA_VISIBLE_DEVICES=0
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

# $1:   exp_name
# $2:   train/evaluate
# $3:   last/coco
if [ $# -lt 3 ]
then
    echo "参数数量不够"
else
    python3.4 $SHELL_FOLDER/../config/$1/run.py $2 --weights=$3
fi
