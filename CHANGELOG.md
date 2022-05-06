# General_Detection
# TODO, 给每一个文件夹下添加 __init__.py 文件，让import更简洁

## 2022年5月6日09:40:55
1.  将2022年1月8日20:40:35实现的功能升级；
    基本上抛弃了argparse的使用，所有参数全部使用yaml文件控制，使用yacs包实现。
    Commit: a0459da4aa7e9ccb28827ab23412dab68b32cc51

## 2022年1月17日16:32:12
1. 新增了lr调整策略，MultiStep_LR, lr阶段下降策略，位置在utils/lr_scheduler.py下
2. 优化了构建optimizer和lr_scheduler的代码，更简洁，灵活，实现了自定义构建。
3. 新增了IOU_loss

## 2022年1月16日16:31:07
1. 添加了TensorboardX功能，可以记录lr, AP_50, avg_loss

## 2022年1月8日20:40:35

1. 将'config'文件格式从.py 改成了'yaml'文件。
2. 实现读取yaml配置文件并转换成属性类。

