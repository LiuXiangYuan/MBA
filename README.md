# MBA
This repository is the implementation of our SIGIR 2023 Paper "Improving Implicit Feedback-Based Recommendation through Multi-Behavior Alignment"



## Requirements

- torch == 1.10.1
- scipy == 1.7.1
- numpy
- pandas
- python3



## Pre-train

We first need to pre-train two models on multi-behavior data.

Run the shell below to pre-train two MF models on Beibei or Taobao dataset (set `--dataset "taobao"` to pre-train two models on Taobao dataset). 

```shell
python main.py --dataset "beibei" --train_method "pre" --model "MF" --lambda0 1e-4 --pretrain_model "MF" --pretrain_early_stop_rounds 20
```



Run the shell below to pre-train two LightGCN models on Beibei or Taobao dataset (set `--dataset "taobao"` to pre-train two models on Taobao dataset). 

```shell
python main.py --dataset "beibei" --train_method "pre" --model "lgn" --lambda0 1e-6 --pretrain_model "lgn" --pretrain_early_stop_rounds 20
```



## MBA-train

After we pre-trained two models on multi-behavior data, we then use them to train our target model with the MBA learning framework.

Run the shell below to train target model.

```shell
python main.py --train_method "mba" --dataset ${dataset} --model ${model_name} --pretrain_model ${model_name} --alpha ${alpha} --C_1 ${C_1} --C_2 ${C_2} --lambda0 ${lambda0}  --beta ${beta}
```

Here we provide a selection of some of the best hyperparameters.


| dataset | model_name | $\alpha$ | $C_1$ | $C_2$ | lambda0 |   $\beta$   |
| :-----: | :--------: | :------: | :---: | :---: | :-----: | :---------: |
| beibei  |     MF     |    10    |  100  |  100  |  1e-4   | 0.7(or 0.8) |
| beibei  |    lgn     |    10    |  10   |  10   |  1e-6   |     1.0     |
| taobao  |     MF     |    1     |   1   |   1   |  1e-4   |     0.8     |
| taobao  |    lgn     |   1000   | 1000  |   1   |  1e-6   | 0.8(or 0.7) |

**Note: MBD dataset is still under approval as it is the company property.**



## Contact

If you have any questions for our paper or codes, please send an email to [chrisxiangyuan@gmail.com](mailto:chrisxiangyuan@gmail.com), or raise an issue to me.



## Acknowledgement

Our code are developed based on [DeCA](https://github.com/wangyu-ustc/DeCA)

Any scientific publications that use our codes should cite our paper as the reference.