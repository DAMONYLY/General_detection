# nohup python -u train.py --device 0,1 --b 40 > test.log 2>&1 &
# nohup python -u train.py --b 30 --device 4,5  --save_path ./results/shufflenetv2_1.0_9 > shufflenetv2_1.0_9.log 2>&1 &
# nohup python -u train.py --weight_path darknet53_448.weights --save_path ./results/darknet_test_33 --b 20 --device 6,7  > darknet_test_33.log 2>&1 &
nohup python -u train.py --dataset_path ./dataset/coco \
--config ./config/test_coco.yaml --save_path shufflenetv2_1.0_coco2 --b 40 --device 4,5,6,7  > shufflenetv2_1.0_coco2.log 2>&1 &