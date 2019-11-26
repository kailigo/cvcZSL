# Rethinking Zero-Shot Learning: A Conditional Visual Classification Perspective (Under Construction)
PyTorch code for the following paper

[Kai Li](http://kailigo.github.io/), [Martin Renqiang Min](http://www.cs.toronto.edu/~cuty/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/). "Rethinking Zero-Shot Learning: A Conditional Visual Classification Perspective", ICCV, 2019. [[pdf](https://arxiv.org/pdf/1909.05995.pdf)]

## Introduction
Zero-shot learning (ZSL) aims to recognize instances of unseen classes solely based on the semantic descriptions of the classes. Existing algorithms usually formulate it as a semantic-visual correspondence problem, by learning mappings from one feature space to the other. Despite being reasonable, previous approaches essentially discard the highly precious discriminative power of visual features in an implicit way, and thus produce undesirable results.  We instead reformulate ZSL as a conditioned visual classification problem, i.e., classifying visual features based on the classifiers learned from the semantic descriptions. With this reformulation, we develop algorithms targeting various ZSL settings: For the conventional setting, we propose to train a deep neural network that directly generates visual feature classifiers from the semantic attributes with an episode-based training scheme; For the generalized setting, we concatenate the learned highly discriminative classifiers for seen classes and the generated classifiers for unseen classes to classify visual features of all classes; For the transductive setting, we exploit unlabeled data to effectively calibrate the classifier generator using a novel learning-without-forgetting self-training mechanism and guide the process by a robust generalized cross-entropy loss. Extensive experiments show that our proposed algorithms significantly outperform state-of-the-art methods by large margins on most benchmark datasets in all the ZSL settings.

## Environment 
We recommended the following dependencies.

* Python 3.5 
* PyTorch (0.4.1)



## Data

Download data from [here](http://www.robots.ox.ac.uk/~lz/DEM_cvpr2017/data.zip) and unzip it `unzip data.zip`.

## Training
### Inductive setting
```bash
python train.py --dataset AwA1 --ways 16 --shots 4 --lr 1e-5 --opt_decay 1e-4 --step_size 500 --log_file eps_lr5_opt4_ss500_w16_s4 --model_file lr5_opt4_ss500_w16_s4.pt
```
### Transductive setting
```bash
python train_transductive.py --dataset AwA1 --ways 16 --shot 1 --lr 1e-4 --opt_decay 1e-5 --step_size 200 --loss_q 5e-1 --trans_model_name trans_s1w16_lr4_opt5_ss200_q5e1_r --log_file trans_s1w16_lr4_opt5_ss200_q5e1_r
```

## Evaluate trained models
### Inductive setting
```bash
python test.py --dataset AwA1 --model_file awa1.pt
```
### Transductive setting
```bash
python train_transductive.py --dataset AwA1 --ways 16 --shot 1 --lr 1e-4 --opt_decay 1e-5 --step_size 200 --loss_q 5e-1 --trans_model_name trans_s1w16_lr4_opt5_ss200_q5e1_r --log_file trans_s1w16_lr4_opt5_ss200_q5e1_r
```


## Reference

If you found this code useful, please cite the following paper:

	@inproceedings{li2019rethinking,
	  title={Rethinking Zero-Shot Learning: A Conditional Visual Classification Perspective},
	  author={Li, Kai and Min, Martin Renqiang and Fu, Yun},
	  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
	  pages={3583--3592},
	  year={2019}
	}

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

