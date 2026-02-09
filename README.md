# LELEM team: Molecular Graph Captioning

This repository contains the code for the Molecular Graph Captioning competition entry by the LELEM team. The competition was hosted as a part of the ALTEGRAD 2025 class at MVA, ENS Paris-Saclay.

## Paper
You can check our paper report [here](https://github.com/vujadinovicn/molecular-graph-captioning/tree/main/docs).

## Getting started
You can create a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```
Note that we used `Python 3.10.19` for this project, so make sure to use a compatible version.

## Training
To train the model, make sure you have the pickle files provided in the `data/` directory. You can then run the training script as follows:

### Stage 1: Pre-training graph model using contrastive learning

```bash
python train_contrastive.py \
    --graphs_path "directory containing data.pkl" \
    --llm_model_id "hf model name"
```
There are several other flags related to temperature, learning rates, batch size, model & architecture, etc. You can find the complete list of arguments by running:

```bash
python train_contrastive.py --help
```

### Stage 2: Prefix tunig of LLM with graph features

```bash
python train_llama.py \
    --graphs_path "directory containing data.pkl" \
    --load_checkpoint_path "path to pre-trained graph model" \
    --llm_model_id "hf model name" \
    --batch_size 8 
```

For using bigger models, you might want to use PEFT using the following flags:
```bash
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 
```

Again, find the complete list of arguments by running:

```bash
python train_llama.py --help
```

## Generation

To generate captions for new molecular graphs using a trained model, you can use the `generate_captions.py` script:

```bash
python generate_captions.py \
  --main_ckpt_path "path to .pt file saved during training" \
  --graph_ckpt_path "path to the pre-trained graph model" \
  --data_path "path to directory containing pickle files" \
  --split "train or test or validation" \
  --output_file "output.json"
```

Note that this will output a JSON file with the following format:
```json
[
  {
    "graph_id": "unique identifier for the graph",
    "generated": "generated caption for the graph",
    "ground_truth": "ground truth caption for the graph if available"
  },
.
.
.

]
```
