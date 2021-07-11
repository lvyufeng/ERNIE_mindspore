
# 目录

<!-- TOC -->

- [目录](#目录)
- [ERNIE概述](#ernie概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [预训练](#预训练)
        - [微调与评估](#微调与评估)
    - [选项及参数](#选项及参数)
        - [选项](#选项)
        - [参数](#参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器上运行](#ascend处理器上运行)
        - [分布式训练](#分布式训练)
            - [Ascend处理器上运行](#ascend处理器上运行-1)
    - [微调过程](#微调过程)
        - [用法](#用法-1)
            - [迁移Paddle预训练权重](#迁移paddle预训练权重)
            - [Ascend处理器上运行单卡微调](#ascend处理器上运行单卡微调)
            - [Ascend处理器上单机多卡微调](#ascend处理器上单机多卡微调)
            - [Ascend处理器上运行微调后的模型评估](#ascend处理器上运行微调后的模型评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果-2)
    - [模型描述](#模型描述)
    - [精度与性能](#精度与性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# ERNIE概述

ERNIE 1.0 通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 BERT 学习原始语言信号，ERNIE 直接对先验语义知识单元进行建模，增强了模型语义表示能力。

[论文](https://arxiv.org/abs/1904.09223): Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, Hua Wu. [ERNIE: Enhanced Representation through Knowledge Integration
](https://arxiv.org/abs/1904.09223). arXiv preprint arXiv:1904.09223.

# 模型架构

Ernie的主干结构为Transformer。对于Ernie_base，Transformer包含12个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。

# 数据集

- 生成下游任务数据集
    - 下载数据集进行微调和评估，如Chnsenticor、CMRC2018、DRCD、MSRA NER、NLPCC DBQA、XNLI等。
    - 将数据集文件从JSON或tsv格式转换为MindRecord格式。

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：

- 在Ascend上运行

```bash
# 单机运行预训练示例
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128

# 分布式运行预训练示例
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json

# 运行微调和评估示例
- 如需运行微调任务，请先准备预训练生成的权重文件（ckpt）。
- 在`finetune_eval_config.py`中设置BERT网络配置和优化器超参。

- 分类任务：在scripts/run_classifier.sh中设置任务相关的超参。
- 运行`bash scripts/run_classifier.py`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_classifier.sh

- NER任务：在scripts/run_ner.sh中设置任务相关的超参。
- 运行`bash scripts/run_ner.py`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_ner.sh

- SQUAD任务：在scripts/run_squad.sh中设置任务相关的超参。
-运行`bash scripts/run_squad.py`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_squad.sh
```

## 脚本说明

## 脚本和样例代码

```shell
.
└─ernie
  ├─README_CN.md
  ├─scripts
    ├─convert_finetune_datasets.sh            # 转换用于微调的JSON或TSV格式数据为MindRecord数据脚本
    ├─convert_pretraining_datasets.sh         # 转换用于预训练的数据为MindRecord数据脚本
    ├─download_datasets.sh                    # 下载微调或预训练数据集脚本
    ├─download_pretrained_models.sh           # 下载预训练模型权重参数脚本
    ├─export.sh                               # 导出模型中间表示脚本，如MindIR
    ├─migrate_pretrained_models.sh            # 在GPU设备上将Paddle预训练权重参数转为MindSpore权重参数脚本
    ├─run_distribute_finetune.sh              # Ascend设备上多卡运行微调任务脚本
    ├─run_finetune_eval.sh                    # Ascend设备上测试微调结果脚本
    ├─run_infer_310.sh                        # Ascend 310设备推理脚本
    ├─run_standalone_finetune.sh              # Ascend设备上单卡运行微调任务脚本
    └─run_standalone_pretrain.sh              # Ascend设备上单卡运行预训练脚本
  ├─src
    ├─__init__.py
    ├─adam.py                                 # 评估过程的测评方法
    ├─assessment_method.py                    # 评估过程的测评方法
    ├─config.py                               # 评估过程的测评方法
    ├─crf.py                                  # 评估过程的测评方法
    ├─dataset.py                              # 评估过程的测评方法
    ├─ernie_for_finetune.py                   # 网络骨干编码
    ├─ernie_for_pretraining.py                # 网络骨干编码
    ├─ernie_model.py                          # 网络骨干编码
    ├─finetune_eval_config.py                 # 数据预处理
    ├─finetune_eval_model.py                  # 网络骨干编码
    ├─pretraining_reader.py                   # 样例处理
    ├─task_reader.py                          # 样例处理
    ├─tokenizer.py                            # 样例处理
    ├─utils.py                                # util函数
  ├─export.py                                 # 训练和评估网络
  ├─run_ernie_classifier.py                   # 分类器任务的微调和评估网络
  ├─run_ernie_ner.py                          # NER任务的微调和评估网络
  └─run_ernie_pretrain.py                     # 预训练网络
```

## 脚本参数

### 微调与评估

```shell
用法：run_ner.py   [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--assessment_method ASSESSMENT_METHOD] [--use_crf USE_CRF]
                    [--device_id N] [--epoch_num N] [--vocab_file_path VOCAB_FILE_PATH]
                    [--label2id_file_path LABEL2ID_FILE_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
选项：
    --device_target                   代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --assessment_method               评估方法，可选项为f1或clue_benchmark
    --use_crf                         是否采用CRF来计算损失，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 BERT模型训练的词汇表
    --label2id_file_path              标注文件，文件中的标注名称必须与原始数据集中所标注的类型名称完全一致
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             如采用f1来评估结果，则为TFRecord文件保存预测；如采用clue_benchmark来评估结果，则为JSON文件保存预测
    --dataset_format                  数据集格式，支持tfrecord和mindrecord格式
    --schema_file_path                模式文件保存路径

用法：run_squad.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--epoch_num N] [--num_class N]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--eval_json_path EVAL_JSON_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       分类数，SQuAD任务通常为2
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 BERT模型训练的词汇表
    --eval_json_path                  保存SQuAD任务开发JSON文件的路径
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存SQuAD训练数据的TFRecord文件，如train1.1.tfrecord
    --eval_data_file_path             用于保存SQuAD预测数据的TFRecord文件，如dev1.1.tfrecord
    --schema_file_path                模式文件保存路径

usage: run_classifier.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                         [--assessment_method ASSESSMENT_METHOD] [--device_id N] [--epoch_num N] [--num_class N]
                         [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                         [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                         [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                         [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                         [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                         [--train_data_file_path TRAIN_DATA_FILE_PATH]
                         [--eval_data_file_path EVAL_DATA_FILE_PATH]
                         [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   任务运行的目标设备，可选项为Ascend或CPU
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --assessment_method               评估方法，可选项为accuracy、f1、mcc、spearman_correlation
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       标注类的数量
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型）
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             用于保存预测数据的TFRecord文件，如dev.tfrecord
    --schema_file_path                模式文件保存路径
```

## 选项及参数

可以在`config.py`和`finetune_eval_config.py`文件中分别配置训练和评估参数。

### 选项

```text
config for lossscale and etc.
    bert_network                    BERT模型版本，可选项为base或nezha，默认为base
    batch_size                      输入数据集的批次大小，默认为16
    loss_scale_value                损失放大初始值，默认为2^32
    scale_factor                    损失放大的更新因子，默认为2
    scale_window                    损失放大的一次更新步数，默认为1000
    optimizer                       网络中采用的优化器，可选项为AdamWerigtDecayDynamicLR、Lamb、或Momentum，默认为Lamb
```

### 参数

```text
数据集和网络参数（预训练/微调/评估）：
    seq_length                      输入序列的长度，默认为128
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同。默认为21136
    hidden_size                     BERT的encoder层数，默认为768
    num_hidden_layers               隐藏层数，默认为12
    num_attention_heads             注意头的数量，默认为12
    intermediate_size               中间层数，默认为3072
    hidden_act                      所采用的激活函数，默认为gelu
    hidden_dropout_prob             BERT输出的随机失活可能性，默认为0.1
    attention_probs_dropout_prob    BERT注意的随机失活可能性，默认为0.1
    max_position_embeddings         序列最大长度，默认为512
    type_vocab_size                 标记类型的词汇表大小，默认为16
    initializer_range               TruncatedNormal的初始值，默认为0.02
    use_relative_positions          是否采用相对位置，可选项为true或false，默认为False
    dtype                           输入的数据类型，可选项为mstype.float16或mstype.float32，默认为mstype.float32
    compute_type                    Bert Transformer的计算类型，可选项为mstype.float16或mstype.float32，默认为mstype.float16

Parameters for optimizer:
    AdamWeightDecay:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率，取值需为正数
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减
    eps                             增加分母，提高小数稳定性

    Lamb:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减

    Momentum:
    learning_rate                   学习率
    momentum                        平均移动动量
```

## 预训练过程

### 用法

#### Ascend处理器上运行

```bash
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128
```

以上命令后台运行，您可以在pretraining_log.txt中查看训练日志。训练结束后，您可以在默认脚本路径下脚本文件夹中找到检查点文件，得到如下损失值：

```text
# grep "epoch" pretraining_log.txt
epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **注意**如果所运行的数据集较大，建议添加一个外部环境变量，确保HCCL不会超时。
>
> ```bash
> export HCCL_CONNECT_TIMEOUT=600
> ```
>
> 将HCCL的超时时间从默认的120秒延长到600秒。
> **注意**若使用的BERT模型较大，保存检查点时可能会出现protobuf错误，可尝试使用下面的环境集。
>
> ```bash
> export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
> ```
<!-- 
### 分布式训练

#### Ascend处理器上运行

```bash
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json
```

以上命令后台运行，您可以在pretraining_log.txt中查看训练日志。训练结束后，您可以在默认LOG*文件夹下找到检查点文件，得到如下损失值：

```text
# grep "epoch" LOG*/pretraining_log.txt
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **注意**训练过程中会根据device_num和处理器总数绑定处理器内核。如果您不希望预训练中绑定处理器内核，请在`scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py`中移除`taskset`相关操作。 -->

## 微调过程

### 用法

#### 迁移Paddle预训练权重

如果没有进行预训练获得模型权重，可以使用百度开源的ERNIE权重，将其转换为MindSpore支持的Checkpoint直接加载进行下游任务微调。

首先下载百度开源ERNIE权重：

```bash
bash scripts/download_pretrained_models.sh
```

下载完成后执行权重迁移脚本：

```bash
bash scripts/migrate_pretrained_models.sh
```

> **注意**： 权重迁移需要同时安装MindSpore和Paddle，由于Paddle不支持Arm环境，本步骤需要在x86环境下运行。权重迁移仅需要两个框架的CPU版本即可完成，可本地完成后上传转换后的Checkpoint使用。

#### Ascend处理器上运行单卡微调

运行以下命令前，确保已设置从Paddle转换或自行训练得到的ERNIE Base的checkpoint。请将检查点路径设置为绝对全路径，例如，/username/pretrain/checkpoint_100_300.ckpt。

```bash
bash scripts/run_standalone_finetune.sh [TASK_TYPE]
# for example: sh run_standalone_finetune.sh msra_ner
# TASK_TYPE including [msra_ner, chnsenticorp]
```

以上命令后台运行，您可以在{task_type}_train_log.txt中查看训练日志。

#### Ascend处理器上单机多卡微调

```bash
bash scripts/run_distribute_finetune.sh [RANK_TABLE_FILE] [TASK_TYPE]
# for example: sh run_distribute_finetune.sh rank_table.json xnli
# TASK_TYPE including [xnli, dbqa, drcd]
```

以上命令后台运行，您可以在{task_type}_train_log.txt中查看训练日志。

> **注意：** `rank_table.json`可以通过`/etc/hccn.conf`获取加速卡IP进行配置。

#### Ascend处理器上运行微调后的模型评估

```bash
bash scripts/run_finetune_eval.sh [TASK_TYPE] 
# for example: sh run_finetune_eval.sh msra_ner
# TASK_TYPE including [msra_ner, chnsenticorp, xnli, dbqa, ]
```

如您选择准确性作为评估方法，可得到如下结果：

```text
acc_num XXX, total_num XXX, accuracy 0.588986
```

如您选择F1作为评估方法，可得到如下结果：

```text
Precision 0.920507
Recall 0.948683
F1 0.920507
```

## 导出mindir模型

```bash
bash export.sh [CKPT_FILE] [EXPORT_PATH] [TASK_TYPE]
# for example: sh sh export.sh /path/ckpt.ckpt /path/ msra_ner
# TASK_TYPE including [msra_ner, chnsenticorp]
```

其中，参数`CKPT_FILE` 是必需的；`EXPORT_FORMAT` 可以在 ["AIR", "MINDIR"]中进行选择后修改`export.sh`, 默认为"MINDIR"。

## 推理过程

### 用法

在执行推理之前，需要通过export.py导出mindir文件。输入数据文件为bin格式。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [LABEL_PATH] [DATA_FILE_PATH] [DATASET_FORMAT] [SCHEMA_PATH] [USE_CRF] [NEED_PREPROCESS] [DEVICE_ID]
```

`NEED_PREPROCESS` 为必选项, 在[y|n]中取值，表示数据是否预处理为bin格式。
`USE_CRF` 为必选项, 在 [true|false]中取值，大小写不敏感。
`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```eval log
F1 0.931243
```

## 模型描述

## 精度与性能

#### 推理性能

##### 命名实体识别任务


| 参数                 | Ascend+Mindspore                        | GPU+Paddle                       |
| -------------------------- | ----------------------------- | ------------------------- |
| 资源                        | Ascend 910；系统 Euler2.8                    | NV SMX2 V100-32G          |
| 上传日期                    | 2021-06-23                    | 2021-06-23                |
| 数据集 | MSRA NER | MSRA NER |
| batch_size          | 32（单卡）                        | 32（单卡）                   |
| Dev准确率 | 95.48% | 95.0% |
| Test准确率 | 94.55% | 93.8% |
| Finetune速度                      | 57.50毫秒/步                              |                           |
| 推理模型 | 1.2G（.ckpt文件）              |                           |

##### 情感分析任务

| 参数                 | Ascend+Mindspore                        | GPU+Paddle                       |
| -------------------------- | ----------------------------- | ------------------------- |
| 资源                   | Ascend 910；系统 Euler2.8                    | NV SMX2 V100-32G          |
| 上传日期              | 2021-06-23                    | 2021-06-23                |
| 数据集 | ChnSentiCorp | ChnSentiCorp |
| batch_size          | 32（单卡）                        | 32（单卡）                   |
| Dev准确率 | 94.83% | 95.2% |
| Test准确率 | 96.08% | 95.4% |
| Finetune速度                      | 57.50毫秒/步                              |                           |
| 推理模型 | 1.2G（.ckpt文件）              |                           |

##### 自然语言接口

| 参数                 | Ascend+Mindspore                        | GPU+Paddle                       |
| -------------------------- | ----------------------------- | ------------------------- |
| 资源                   | Ascend 910；系统 Euler2.8                    | NV SMX2 V100-32G          |
| 上传日期              | 2021-06-23                    | 2021-06-23                |
| 数据集 | XNLI | XNLI |
| batch_size          | 32（单卡）                        | 32（单卡）                   |
| Dev准确率 | 79.1% | 79.9% |
| Test准确率 | 78.4% | 78.4% |
| Finetune速度                      | 495.71毫秒/步                              |                           |
| 推理模型 | 1.2G（.ckpt文件）              |                           |

##### 问答

| 参数                 | Ascend+Mindspore                        | GPU+Paddle                       |
| -------------------------- | ----------------------------- | ------------------------- |
| 资源                   | Ascend 910；系统 Euler2.8                    | NV SMX2 V100-32G          |
| 上传日期              | 2021-06-23                    | 2021-06-23                |
| 数据集 | DBQA | DBQA |
| batch_size          | 32（单卡）                        | 32（单卡）                   |
| Dev准确率 | 80.5% | 82.3% |
| Test准确率 | 82.6% | 82.7% |
| Finetune速度                      | 137.08毫秒/步                              |                           |
| 推理模型 | 1.2G（.ckpt文件）              |                           |

##### 阅读理解

DRCD

| 参数                 | Ascend+Mindspore                        | GPU+Paddle                       |
| -------------------------- | ----------------------------- | ------------------------- |
| 资源                   | Ascend 910；系统 Euler2.8                    | NV SMX2 V100-32G          |
| 上传日期              | 2021-06-23                    | 2021-06-23                |
| 数据集 | ChnSentiCorp | ChnSentiCorp |
| batch_size          | 32（单卡）                        | 32（单卡）                   |
| Dev准确率 | 94.83% | 95.2% |
| Test准确率 | 96.08% | 95.4% |
| Finetune速度                      | 57.50毫秒/步                              |                           |
| 推理模型 | 1.2G（.ckpt文件）              |                           |


CMRC2018

| 参数                 | Ascend+Mindspore                        | GPU+Paddle                       |
| -------------------------- | ----------------------------- | ------------------------- |
| 资源                   | Ascend 910；系统 Euler2.8                    | NV SMX2 V100-32G          |
| 上传日期              | 2021-06-23                    | 2021-06-23                |
| 数据集 | ChnSentiCorp | ChnSentiCorp |
| batch_size          | 32（单卡）                        | 32（单卡）                   |
| Dev准确率 | 94.83% | 95.2% |
| Test准确率 | 96.08% | 95.4% |
| Finetune速度                      | 57.50毫秒/步                              |                           |
| 推理模型 | 1.2G（.ckpt文件）              |                           |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
