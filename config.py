# -*- coding: utf-8 -*-
# @File : config.py  # 文件名
# @Author: Runist  # 作者名
# @Time : 2021/12/15 12:14  # 创建时间
# @Software: PyCharm  # 使用的开发软件
# @Brief: 配置文件  # 文件的简要说明

import argparse  # 导入 argparse 模块，用于解析命令行参数
import os  # 导入 os 模块，用于操作系统相关功能


# 创建一个 ArgumentParser 对象，用于处理命令行参数
parser = argparse.ArgumentParser()

# 添加命令行参数
# parser.add_argument('--num_classes', type=int, default=5,
#                     help='Number of classes for the classification task.')  # 分类任务的类别数量
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs for training.')  # 训练的轮数
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training.')  # 每个训练批次的样本数量
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate for the optimizer.')  # 优化器的学习率
parser.add_argument('--lrf', type=float, default=0.01,
                    help='Learning rate factor for adjusting learning rate.')  # 学习率调整因子

# 添加训练和验证数据集目录参数
parser.add_argument('--dataset_dir', type=str,
                    default= "/remote-home/lms/IndianOceanDepthTime.nc",
                    help='The directory containing the train data.')  # 训练数据的目录

parser.add_argument('--depth',type=int,default=4 )


parser.add_argument('--label',type=int,default=3 )

# 添加保存权重和 TensorBoard 日志的目录参数
parser.add_argument('--summary_dir', type=str, default="./1206_cluster",
                    help='The directory of saving weights and tensorboard.')  # 保存模型权重和日志的目录

# 预训练权重路径，如果不想载入则设置为空字符串
parser.add_argument('--weights', type=str, default='',
                    help='Initial weights path.')  # 预训练模型权重的路径

# 是否冻结权重的参数
parser.add_argument('--freeze_layers', type=bool, default=False,
                    help='Whether to freeze the layers during training.')  # 指定是否在训练时冻结模型层

# GPU 设备选择参数
parser.add_argument('--gpu', type=str, default='0,1,2,3',
                    help='Select GPU device.')  # 指定可用的 GPU 设备

# 模型名称参数，指定要训练的 ViT 模型
parser.add_argument('--model', type=str, default='ocean',
                    help='The name of ViT model, select one to train.')  # 选择要训练的 ViT 模型名称

# # 类别名称列表参数
# parser.add_argument('--label_name', type=list, default=[
#     "daisy",
#     "dandelion",
#     "roses",
#     "sunflowers",
#     "tulips"
# ], help='The name of class.')  # 定义每个类别的名称

# 解析命令行参数，将结果存储在 args 变量中
args = parser.parse_args()

args.num_workers = 4
# 设置可见的 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 根据用户输入设置可见的 GPU 设备