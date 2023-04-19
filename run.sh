model_name=MF
dataset=beibei
alpha=1000
C_1=1000
C_2=1
beta=0.7
lambda0=1e-4
lambda1=1e-6
train_method=mba
idx=0

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset ${dataset} \
--alpha ${alpha} \
--C_1 ${C_1} --C_2 ${C_2} \
--train_method ${train_method} --model ${model_name} \
--lambda0 ${lambda0} --lambda1 ${lambda1} \
--pretrain_model ${model_name} \
--beta ${beta} \
--pretrain_early_stop_rounds 20 --idx ${idx} > run.log 2>&1 &
