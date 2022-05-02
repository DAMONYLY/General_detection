# General_Detection

## 2022年1月17日16:32:12
1. 新增了lr调整策略，MultiStep_LR, lr阶段下降策略，位置在utils/lr_scheduler.py下
2. 优化了构建optimizer和lr_scheduler的代码，更简洁，灵活，实现了自定义构建。
3. 新增了IOU_loss

## 2022年1月16日16:31:07
1. 添加了TensorboardX功能，可以记录lr, AP_50, avg_loss

## 2022年1月8日20:40:35

1. 将'config'文件格式从.py 改成了'yaml'文件。
2. 实现读取yaml配置文件并转换成属性类。

