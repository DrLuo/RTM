# Toward real text manipulation detection: New dataset and new solution
This is the official repository of paper ["Toward real text manipulation detection: New dataset and new solution"](https://doi.org/10.1016/j.patcog.2024.110828) (Pattern Recogntion, 2024).

## RTM dataset

The RTM dataset consists of 9,000 text images in total, including 6,000 manually manipulated text images and 3,000 authentic images. The dataset is available at [Google Drive](https://drive.google.com/file/d/11AHZ8ih_kDCFilGceevppcGkKR4vDJD2/view?usp=sharing).

## Evaluation Tool

Before running the srcipt, please make sure the prediction folder is renamed following the format:

{MethodName}_mask

For example: ascformer_mask

```
cd EvalRTM
python run_eval.py --pred_dir ${PRED_FOLDER} --gt_dir ${RTM_GT_FOLDER}
```
We use pqdm to accelerate the evaluating process. The evaluation results will be saved in Json file and shown using PrettyTable.

## ASCFormer

To be released



## TODO
- [x] Release dataset
- [x] Release evaluation code
- [ ] Release model (Soon)

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
