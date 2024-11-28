# Toward real text manipulation detection: New dataset and new solution
This is the official repository of paper ["Toward real text manipulation detection: New dataset and new solution"](https://doi.org/10.1016/j.patcog.2024.110828) (Pattern Recogntion, 2024).

## RTM dataset

The RTM dataset consists of 9,000 text images in total, including 6,000 manually manipulated text images and 3,000 authentic images. The dataset is available at [Google Drive](https://drive.google.com/file/d/11AHZ8ih_kDCFilGceevppcGkKR4vDJD2/view?usp=sharing).

## Evaluation Tool

Before running the srcipt, please make sure the prediction folder is renamed following the format:
`{MethodName}_mask`

For example: `ascformer_mask`

```shell
cd EvalRTM
python run_eval.py --pred_dir ${PRED_FOLDER} --gt_dir ${RTM_GT_FOLDER}
```
We use pqdm to accelerate the evaluating process. The evaluation results will be saved in Json file and shown using PrettyTable.

## ASCFormer

### Installation

This repo depends on This repo depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
Below are quick steps for installation.
Please refer to [MMSegmentation Install Guide](https://mmsegmentation.readthedocs.io/en/dev-1.x/get_started.html) for more detailed instruction.

Python 3.8 + PyTorch 2.0.0 + CUDA 11.8 + mmsegmentation (1.0.0rc6)

```shell
conda create --name rtm python=3.8 -y
conda activate rtm
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -U openmim
mim install "mmengine==0.7.0"
mim install "mmcv==2.0.0"

git clone https://github.com/DrLuo/RTM.git
cd RTM
cd ASCFormer
pip install -r requirements.txt
pip install -v -e .
```

### Prepare dataset

Place the RTM dataset at `./data/ttd/RealTextMan` 

Organize the files as follows

```
|- ./data
   |- ttd
      |- RealTextMan
         |- JPEGImages
         |- SegmentationClass
         |- train.txt
         |- val.txt
         └  test.txt
```


### Training

For distributed training on multiple GPUs, please use

```shell
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

For training on a single GPU, please use

```shell
python ./tools/train.py ${CONFIG_FILE} ${GPU_NUM}
```

For example, we use this script to train the model:

```shell
bash tools/dist_train.sh configs/ascformer_rtm/ascformer_model.pth 2
```


### Inference

For inference on multiple GPUs, please use

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} --mask
```

For inference on single GPU, please use

```shell
python ${CONFIG_FILE} ${CHECKPOINT_FILE} --mask
```



For example, we use this script to inference and evaluate:

```shell
bash tools/dist_test.sh configs/ascformer/ascformer_rtm.py work_dirs/ascformer_rtm/ascformer_model.pth ${NUM_GPUS} --mask
```

### Evaluation

After obtaining the binary masks, please use the evaluation tool of RTM for more detailed evaluation.


### Main Results

|Method|CM|SP|GN|CV|IP|CB|Tamper|All|download|
|-|-|-|-|-|-|-|-|-|-|
|ASC-Former|18.57|32.79|18.89|16.06|27.63|19.35|21.57|19.71|[model](https://drive.google.com/file/d/1xltdrDhqeyDh3TnynXDn0eVZnAKtnlKx/view?usp=sharing)|


## TODO
- [x] Release dataset
- [x] Release evaluation code
- [ ] Release model (On going)

## Citation
Please cite the following paper when using the RTM dataset or this repo.

```
@article{luo2024toward,
  title={Toward real text manipulation detection: New dataset and new solution},
  author={Luo, Dongliang and Liu, Yuliang and Yang, Rui and Liu, Xianjin and Zeng, Jishen and Zhou, Yu and Bai, Xiang},
  journal={Pattern Recognition},
  pages={110828},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgement

This repo is based on [MMSegmentation 1.0.0rc6](https://github.com/open-mmlab/mmsegmentation). We appreciate this wonderful open-source toolbox.