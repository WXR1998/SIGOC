SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

# $1:   exp_name
# $2:   train/evaluate
# $3:   last/coco
if [ $# -lt 4 ]
then
    echo "参数数量不够"
else
    export CUDA_VISIBLE_DEVICES=$4
    python3.4 $SHELL_FOLDER/../config/$1/run.py $2 --weights=$3 --limit=1000
fi
