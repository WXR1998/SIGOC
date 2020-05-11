SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

# $1:   exp_name
# $2:   graphics card
if [ $# -lt 2 ]
then
    echo "参数数量不够"
else
    export CUDA_VISIBLE_DEVICES=$2
    python3.4 $SHELL_FOLDER/../config/$1/run.py train --weights=coco
    python3.4 $SHELL_FOLDER/../config/$1/run.py evaluate --weights=last
fi
