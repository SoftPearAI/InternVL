conda create --name internvl_inf python=3.10

conda create --name internvl2 python=3.10

conda activate internvl2

pip install lmdeploy

nohup ./shell/internlm2_20b_dynamic/my_finetune3.sh > train.logs &