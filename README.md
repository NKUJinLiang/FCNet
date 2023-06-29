# FCNet

This project is the official pytorch implementation of the paper [FCNet: A Convolutional Neural Network for Arbitrary-Length Exposure Estimation](https://arxiv.org/abs/2203.03624).

## prepare data

1. First download [Training](https://ln2.sync.com/dl/141f68cf0/mrt3jtm9-ywbdrvtw-avba76t4-w6fw8fzj)|[Validation](https://ln2.sync.com/dl/49a6738c0/3m3imxpe-w6eqiczn-vripaqcf-jpswtcfr)|[Testing](https://ln2.sync.com/dl/098a6c5e0/cienw23w-usca2rgh-u5fxiex-q7vydzkp) from the [MSEC github repository](https://github.com/mahmoudnafifi/Exposure_Correction);
2. Place the dataset in the root directory of the project.

## train

The training pipeline will be organized within a few weeks.

## test

The trained checkpoint has been uploaded to [snapshot](https://github.com/NKUJinLiang/FCNet/tree/main/snapshots).

For fusion evaluation, run:
```
python fusiontest.py --exposure under
python fusiontest.py --exposure over
python fusiontest.py --exposure all
```

For single image exposure correction evaluation, run:
```
python correctiontest.py
```
