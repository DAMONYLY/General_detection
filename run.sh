# nohup python -u train.py --device 0,1 --b 40 > test.log 2>&1 &
nohup python -u train.py --save_path ./results/06 --save_model --b 60 --device 0,1  > test1.log 2>&1 &
