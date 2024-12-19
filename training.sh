#!/bin/bash

problems=("FCMCNF" "GISP" "WPMS")


echo "10 percent training..."

for problem in "${problems[@]}"; do
    # 替换 problem 的命令并执行
    python learning/train.py -problem $problem -data_type 1
    # 测试集运行
    python main.py -problem $problem -n_cpu 8 -data_partition test

    # 转移集运行
    python main.py -problem $problem -n_cpu 8 -data_partition transfer
done

mv stats stats_10

echo "Full training..."
for problem in "${problems[@]}"; do
    # 替换 problem 的命令并执行
    python learning/train.py -problem $problem -data_type 0
    # 测试集运行
    python main.py -problem $problem -n_cpu 8 -data_partition test

    # 转移集运行
    python main.py -problem $problem -n_cpu 8 -data_partition transfer
done

mv stats stats_100
