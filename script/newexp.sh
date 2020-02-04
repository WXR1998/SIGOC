#!/bin/zsh
if [ $# = 2 ]
then
    SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
    if [ -d $SHELL_FOLDER/../config/$2 ]
    then
        echo "新文件夹已存在"
    else
        cp -r $SHELL_FOLDER/../config/$1 $SHELL_FOLDER/../config/$2
    fi
else
    echo "请输入 [旧文件夹名] [新文件夹名]。"
fi