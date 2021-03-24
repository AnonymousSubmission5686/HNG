
# HNG
## (Our code is based on the [OpenUnReID](https://github.com/open-mmlab/OpenUnReID) toolbox.)

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Test
We provide models trained on DukeMTMC-ReID for testing. 
You can download them([baseline+ours_duke.pth](https://drive.google.com/file/d/1JA_vFjcLB_AwxV25aH3MyHA3lH_Ea7K6/view?usp=sharing), [spcl+ours_duke.pth](https://drive.google.com/file/d/12Jx466C_GcF-ulzRs_WBc4eNC2s6fA9t/view?usp=sharing)) from google drive, and place in `logs`.

+ Testing baseline+ours on DukeMTMC-ReID:
```shell
bash dist_test.sh ../logs/baseline+ours_duke.pth strong_baseline/config.yaml
```

+ Testing spcl+ours on DukeMTMC-ReID:
```shell
bash dist_test.sh ../logs/spcl+ours_duke.pth SpCL/config.yaml
```

## Train
We provide warmup models([baseline_warmup_duke.pth](https://drive.google.com/file/d/1q7zXZ4w7iYfK_Ew7_fQ4N1Jakk61HcVz/view?usp=sharing), [spcl_warmup_duke.pth](https://drive.google.com/file/d/1fo1-A_Aokp9ollq1BFXMwQ7sdacbCmUt/view?usp=sharing), [warmup_generator_duke.pth](https://drive.google.com/file/d/1HmQKMTiiRN9MWeNBbtV3bRj5brPbDbd7/view?usp=sharing), [warmup_discriminator_duke.pth](https://drive.google.com/file/d/1OX0njSr5_OR9_yqkfwH3eGRW86qqShxG/view?usp=sharing)) for simplifing the training procedure, download and put them in `logs`.

+ Training baseline+ours:
```shell
bash dist_train.sh strong_baseline/config.yaml duke/baseline+ours
```

+ Training spcl+ours:
```shell
bash dist_train.sh SpCL/config.yaml duke/spcl+ours
```


