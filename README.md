# Molecular Graph Captioning

The solution to the **ALTEGRAD 2025** molecular graph captioning challenge (MVA, ENS Paris-Saclay): given a molecule as a graph of atoms and bonds, write the natural-language description a chemist would give it.

We connect a **GNN encoder** to an instruction-tuned **LLaMA** in two stages: contrastive alignment first, then prefix-tuning of the LLM on the graph embeddings. This repository contains the full pipeline, plus the variants we compared along the way (model size, tuning strategy, re-ranking).

**Report**: [docs/paper.pdf](docs/paper.pdf)

---

## Table of Contents

- [Overview](#overview)
- [What We Implemented](#what-we-implemented)
- [Approaches We Tried](#approaches-we-tried)
- [Results](#results)
- [Setup](#setup)
- [Training](#training)
- [Generation](#generation)
- [Repository Structure](#repository-structure)

---

## Overview

**Task.** Given a molecular graph (atoms as nodes, bonds as edges, with categorical features), generate a free-text caption such as *"The molecule is a member of the class of benzoic acids..."*.

**Our approach.** Two training stages, shown below:

1. **Contrastive alignment.** A GINE encoder and a projector are trained so that graph embeddings match caption embeddings taken from the 16th block of a frozen LLaMA (InfoNCE on mean- and std-pooled features).
2. **Prefix-tuning.** The aligned node embeddings are injected into the LLaMA prompt, one token per atom, and the model is instruction-tuned to write the caption.

![Two-stage training](figures/pipeline.png)

---

## What We Implemented

| Component | Description |
|---|---|
| GINE graph encoder | Atom/bond embedding tables, 5 edge-aware GINE layers, per-node outputs |
| Projector | 2-layer MLP with GELU and LayerNorm, maps nodes to the LLM embedding space |
| Contrastive alignment | Symmetric InfoNCE on mean- and std-pooled graph and LLaMA layer-16 features|
| Prefix-tuning | Each atom is a reserved special token whose embedding is replaced by its projected node embedding |
| Instruction tuning | System prompt with random in-context caption examples, loss on the answer only |
| LoRA | Optional parameter-efficient tuning used for the 8B model |
| Generation | Beam search producing a JSON of generated and reference captions |
| Re-ranking | Selecting among candidate captions by graph-caption similarity or with an LLM judge |

---

## Approaches We Tried

| Axis | Variants |
|---|---|
| Graph model | Trained from scratch vs. contrastively pre-trained and frozen vs. pre-trained and fine-tuned |
| LLM tuning | Instruction tuning (loss on answer) vs. supervised fine-tuning (loss on whole sequence) |
| LLM size | Llama-3.2-1B, Llama-3.2-3B, Llama-3-8B |
| LLM training | Full fine-tuning for the 1B and 3B models, LoRA for the 8B model |
| Post-processing | None vs. re-ranking by graph–caption similarity vs. re-ranking with a Llama-3-8B judge |

---

## Results

Validation scores of all training methods, evaluated with two BERTScore backbones (Composite = mean of BLEU and rescaled BERTScore). IT = instruction tuning, SFT = supervised fine-tuning.

| LLM | Graph model | Frozen G | LLM tuning | Eval model | BLEU | BERTScore | Composite |
|---|---|:---:|:---:|---|---:|---:|---:|
| 1B | From scratch | ✗ | IT | RoBERTa-base | 12.57 | 24.76 | 18.67 |
| | | | | ChemBERTa-ZINC | | 96.21 | 54.39 |
| 1B | Pre-trained | ✓ | IT | RoBERTa-base | 14.30 | 24.54 | 19.42 |
| | | | | ChemBERTa-ZINC | | 96.43 | 55.37 |
| 1B | Pre-trained | ✗ | IT | RoBERTa-base | 40.03 | 56.32 | 48.17 |
| | | | | ChemBERTa-ZINC | | 97.44 | 68.74 |
| 1B | Pre-trained | ✓ | SFT | RoBERTa-base | 13.70 | 24.82 | 18.48 |
| | | | | ChemBERTa-ZINC | | 96.15 | 54.14 |
| 3B | Pre-trained | ✓ | IT | RoBERTa-base | 39.85 | 56.47 | 48.16 |
| | | | | ChemBERTa-ZINC | | 97.46 | 68.66 |
| 8B (LoRA) | Pre-trained | ✓ | IT | RoBERTa-base | 34.67 | 52.86 | 43.76 |
| | | | | ChemBERTa-ZINC | | 97.27 | 65.97 |

Best public Kaggle score is achieved for **Llama-3.2-1B, pre-trained GNN, instruction tuning** at **0.63748**.

---

## Setup

### Requirements

- Python 3.10 (we used 3.10.19)
- CUDA-capable GPU
- A Hugging Face account with access to the LLaMA models

### Installation

```bash
pip install -r requirements.txt
```


---

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

---

## Repository Structure

```
molecular-graph-captioning/
  train_contrastive.py       - stage 1: contrastive alignment of graph encoder and projector
  train_llama.py             - stage 2: prefix-tuning of the LLM (full or LoRA)
  generate_captions.py       - beam-search caption generation
  models/
    graph_encoder.py         - atom/bond encoder and GINE layers
    projector.py             - MLP projector
    readout.py               - mean / std pooling of embeddings
    model.py                 - graph-conditioned LLM
  losses/loss.py             - InfoNCE loss
  data/                      - dataset, collater, prompt construction
  utils/plotting.py          - plotting helpers
  figures/                   - figures used in this README
  docs/paper.pdf             - project report
```

---

**Authors:** Kshitij Ambilduke, Nemanja Vujadinović and Mohammed El Hassan Ayoubi -- MVA, ENS Paris-Saclay.
