# General_Detection 更新信息

## v1.0 (2022年5月28日)
- 舍弃 DataParallel(DP)方法进行多卡训练
- 支持分布式训练 DistributedDataParallel(DDP)功能，训练加速50%
- 目前不支持cpu训练，后续解决
- 新增config.yaml使用说明文档
## 2022年5月19日16:02:31
- 支持logger存储训练信息功能
- Commit: a243a0a5220c2ba75ede7e620754c9a7ec2a5c85
## 2022年5月18日22:27:14
- 支持dataloader的专用函数，进一步优化train.py下的代码整洁
- 支持Data_prefetcher功能，加速数据加载，减少训练时间
- Commit: c0f1a4cb8d56af3a98a034a1aa4a8ba4878dfbc1

## 2022年5月17日22:47:21
- 支持ATSS_assigner部分代码
- Commit: f11a575fb6a77974af0f25b508fdc5fd978a3e98

## 2022年5月15日22:45:04
- 优化loss_calculater部分代码结构
- 结构更加标准，为了方便后续的ATSS_assigner的更新
- Commit: 3c5a4156c152c0ebc2a99281a301145d2bf0f8e2

## 2022年5月14日22:40:01
- 支持训练yaml文件保存功能
- 支持数据集图片检查功能
- 可以检查data_agument之后的图片和标注
- Commit: 3efd68e1fa5c0660907df0ea65af1d6f496435b4

## 2022年5月14日17:40:01
- 删除了部分冗余代码
- 支持模型初始化的功能，现在基本上MAP能对应的上mmdetection
- 支持训练yaml文件保存功能
- Commit: 69fc45a21a20864009cbd0df44aae127c41228db

## 2022年5月8日22:47:44
- 升级了COCO_evaluate的部分
- 现在支持test时batchsize为任意， 同时结构更加清晰明了
- 删除了部分冗余代码
- Commit: 9a151bc0d794e27a13b83debc9f8ec696f5aa592

## 2022年5月8日19:42:30
- 将原来数据增强（包括resize）部分进行改进
- 改用pipeline的形式，可以通过yaml文件控制数据增强策略
- 目前只完成了将原图resize到指定input_size功能，以及hsv，flip功能
- 后续应当实现nomerlize，mixup，mosanic...等等，以及train和val分开
- Commit: e023efc1a65a7fb4a701c754d47af8df9789278a

## 2022年5月6日09:40:55
- 将2022年1月8日20:40:35实现的功能升级
- 基本上抛弃了argparse的使用，所有参数全部使用yaml文件控制，使用yacs包实现
- Commit: a0459da4aa7e9ccb28827ab23412dab68b32cc51

## 2022年1月17日16:32:12
- 支持了lr调整策略，MultiStep_LR, lr阶段下降策略，位置在utils/lr_scheduler.py下
- 优化了构建optimizer和lr_scheduler的代码，更简洁，灵活，实现了自定义构建
- 支持了IOU_loss

## 2022年1月16日16:31:07
- 添加了TensorboardX功能，可以记录lr, AP_50, avg_loss

## 2022年1月8日20:40:35
- 将'config'文件格式从.py 改成了'yaml'文件
- 实现读取yaml配置文件并转换成属性类