# WillAIm

## Requirements

To create and activate the conda environment named `ldmpoem` using:

```
conda env create -f environment.yaml
conda activate ldmpoem
```

or 

```
PIP_EXISTS_ACTION=w conda env create -f environment.yaml
conda activate ldmpoem
```

## Main execution paths

### 1) data downloading + generation

The first step is crawling

```
python scripts/poetry_foundation_scrape.py
```

Then we can use the following to generate the training set (note limits on the for loop is due to having had to start stop multiple times and so starting later)

```
python scripts/poem2img.py
```

We can generate the test set using:

```
python scripts/poem2img_test.py
```

Notice in the scripts directory there are many cleaning scripts, this was because of having to join training sets and issues with the initial encodings which introduced grammar errors.

### 2) training your baselines and 3) training your experiments

We can use, there was experimentations with different batch-sizes and other hyperparameters but the ones present in this are the official and final parameters.

```
#MLP model
python train.py 1 ViT-B_32_train_nws.pkl
python train.py 1 ViT-B_32_train.pkl

# Transformer model
python train.py 2 ViT-B_32_train_nws.pkl
python train.py 2 ViT-B_32_train.pkl
```

The GPT-3 code is used straight up into the test_gpt3 and the Multi-M I downloaded the latest weights from their own repository to ensure faithful comparisons

### 4) evaluation your model output

To evaluate the model output, we can perform the following commands:

```
python test_gpt.py
python test_gpt3.py
python test_multim.py
python test_transformer.py
```

For the first and lest, that are testing the GPT-2 + MLP and GPT-2 + Transformer model, I manually changed the paths to test the NWS and Normal datasets. However, the embeddings and tokens can be found under.

```
data/ViT-B_32_train_nws
data/ViT-B_32_train
```

The above python codes will output a score_final.json, which is available under (respectively):

```
models/gpt_mlp/scores/score_final.json
models/gpt_mlp/scores/score_nws_final.json
models/gpt3/scores/score_final.json
models/m_adv/scores/score_final.json
models/gpt_transformer/scores/score_final.json
models/gpt_transformer/scores/score_nws_final.json
```