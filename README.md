# Regression Under Human Assistance

This is a repository containing code and data for the paper:

> A. De, P. Koley, N. Ganguly and M. Gomez-Rodriguez. _Regression Under Human Assistance._ AAAI Conference on Artificial Intelligence , February, 2020. 

and its extended version:

> A. De, N. Okati, P. Koley, N. Ganguly and M. Gomez-Rodriguez. _Regression Under Human Assistance._ arXiv preprint arXiv:1909.02963., 2021. 

The extended version of the paper is available [here](https://arxiv.org/pdf/1909.02963.pdf).


## Pre-requisites

This code depends on the following packages:

 1. `numpy`
 2. `pillow`
 3. `matplotlib`
 4. `sklearn`
 


## Code structure

 - `algorithms.py` contains the proposed greedy algorithm and the baselines mentioned in the paper.
 - `train.py` trains greedy and baseline algorithms on given dataset.
 - `generate_human_error.py` generates human error for the real datasets.
 - `generate_synthetic.py` generates synthetic datasets.
 - `test.py` tests greedy and baseline algorithms on given dataset and plots the figures in the paper.
 - `preprocess_text_data.py` preprocesses hatespeech dataset.
 - `preprocess_image_data.py` preprocesses image datasets.
 - `/Results` training results for each dataset are saved in this folder.
 - `/plots` generated figures will be saved in this folder.
 - `/data` preprocessed real datasets and generated synthetic datasets are saved in this folder.


## Execution

#### Run the algorithms:
`python train.py [name of datasets that you wish to train]`

For example: `python train.py messidor sigmoid Ugauss`

Available datasets : `['messidor', 'stare5', 'stare11', 'hatespeech','Umessidor', 'Ustare5', 'Ustare11' ,'sigmoid', 'Usigmoid', 'Wsigmoid', 'gauss', 'Ugauss', 'Wgauss']`

- The datasets starting with 'U' are used to generate the U-shaped figures in the paper.

- The datasets starting with 'W' are used to monitor the solution w of the Greedy algorithm. (figure 1)

All the default parameters are set based on the paper. You can change them in the code.

The results will be saved in `/Results` folder. The results corresponding to all datasets are already generated and saved in `/Results`.

#### Test and generate the figures:
`python test.py [name of datasets that you wish to test]`

For example:
`python test.py messidor sigmoid Ugauss`

The figures will be saved in `/data` folder.


#### Regenerate synthetic data:
**There is no need to run this script if you want to use the same synthetic data as mentioned in the paper**, but, if you wish to generate synthetic datasets with new settings you can modify the `generate_synthetic.py` script and then run:

`python generate_synthetic.py [name of the synthetic datasets]`

For example:
`python generate_synthetic.py sigmoid gauss`

Synthetic datasets : `['sigmoid', 'gauss', 'Usigmoid', 'Ugauss', 'Wsigmoid', 'Wgauss']`

and then train and test to see the new results.


#### Change the human error for the real datasets:
Currently we use Dirichlet prior to generate human error for the image datasets as mentioned in the paper. **There is no need to run this script if you want to use the same human error**, but, if you wish to change it or try other methods you may modify `generate_human_error.py` script and then run:

`python generate_human_error.py [name of image datasets]`


For example:
`python generate_human_error.py messidor stare5 Umessidor`

image datasets are : `['messidor', 'Umessidor', 'stare5', 'Ustare5', 'stare11', 'Ustare11']`

For the datasets which start with 'U', you can use the script to modify the low and high human error values.

and then train and test to see the new results.

----

## Pre-processing

The datasets are preprocessed and saved in `data` folder and **there is no need to download them again**, but, if you wish to change the preprocessing or feature extraction method, you may download the [Messidor](http://www.adcis.net/en/third-party/messidor/), [Stare](https://cecas.clemson.edu/~ahoover/stare/), and [Hatespeech](https://github.com/t-davidson/hate-speech-and-offensive-language) datasets and use `preprocess_image_data.py and preprocess_text_data.py` to preprocess them. You will also need [Resnet](https://github.com/KaimingHe/deep-residual-networks) to generate feature vectors of the image datasets.


## Plots

See the notebook `plots.ipynb`.
