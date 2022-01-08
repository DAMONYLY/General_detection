# nohup python -u train.py --device 0,1 --b 40 > test.log 2>&1 &
nohup python -u train.py --b 30 --device 6,7 --save_path ./results/resnet50   > resnet50.log 2>&1 &
# nohup python -u train.py --weight_path darknet53_448.weights --save_path ./results/darknet_test_2 --b 40 --device 4,5  > darknet_test_2.log 2>&1 &