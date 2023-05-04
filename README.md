# EmpSOA

> The official implementation for the Findings of ACL 2023 paper *Don't Lose Yourself! Empathetic Response Generation via Explicit Self-Other Awareness*.

<img src="https://img.shields.io/badge/Venue-ACL--23-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">

## Requirements
* Python 3.7
* PyTorch 1.8.2
* Transformers 4.12.3
* CUDA 11.1

## Preparation

Download  [**Pretrained GloVe Embeddings**](http://nlp.stanford.edu/data/glove.6B.zip) and save it in `/vectors`.

The processed dataset and generated commonsense knowledge can be downloaded from (https://drive.google.com/drive/folders/1zr6xlD9ryCmVx-P5REfO1Run4aT069aT?usp=share_link). Then move it to `/data`.

We have tried two ways to use the commonsense knowledge: hidden features and decoded texts. And the former one can achieve better results. \
If you want to generate the hidden features: 
```sh
cd comet-atomic-2020/models/comet_atomic2020_bart
```
Download the pretrained COMET model in `download_model.sh`.

Then run `generate_knowledge.py` to get commonsense knowledge features.

```sh
python generate_knowledge.py
```

or you can use the decoded commonsense texts following the dataset processing step. And the preprocessed dataset and decoded commonsense texts would be generated after the training script.

## Training

```sh
python main.py --cuda --save_decode [--wo_sog] [--wo_som] [--wo_sod] [--only_user] [--only_agent] [--wo_dis_sel_oth]
```
The extra flags can be used for ablation studies.

## Evaluation

Create a folder `results` and move the obtained results.txt to this folder:

```sh
python src/scripts/evaluate.py 
```

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@article{zhao2022don,
  title={Don't Lose Yourself! Empathetic Response Generation via Explicit Self-Other Awareness},
  author={Zhao, Weixiang and Zhao, Yanyan and Lu, Xin and Qin, Bing},
  journal={arXiv preprint arXiv:2210.03884},
  year={2022}
}
```

## Credits
The code of this repository partly relies on [CEM](https://github.com/Sahandfer/CEM) and I would like to show my sincere gratitude to authors of it.