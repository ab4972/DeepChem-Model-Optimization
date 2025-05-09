# High Performance Machine Learning: Optimizing ChemBERTa with Parameter-Efficient Techniques for Molecular Toxicity Prediction

[![Open in Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/hpml-proj-deepchem/chemberta)

## Introduction

**ChemBERTa** is a domain-specific BERT model pre-trained on 10 million SMILES strings from the ZINC15 database. Developed by [Chithrananda et al.](https://arxiv.org/abs/2010.09885), it achieves state-of-the-art performance on chemical property prediction tasks. This project optimizes ChemBERTa for the ClinTox toxicity prediction task using parameter-efficient techniques.


**Key Techniques Applied:**
- **LoRA/AdaLoRA**: Low-Rank Adaptation for efficient fine-tuning
- **Quantization**: Post-Training Quantization (PTQ) for model compression
- **torch.compile**: Graph optimizations for faster inference
- **KD-LoRA**: Knowledge Distillation to smaller LoRA-adapted models

## Project Structure

```
deepchem-model-optimization/
├── chemberta_lora_adalora.ipynb # LoRA and AdaLoRA adapted ChemBERTa with hyperparameter tuning
├── Compiled_LoRA.ipynb # Optimized vs base model speeds
├── KDLoRA.ipynb # Knowledge Distillation + LoRA implementation
├── Quantized_LoRA.ipynb # PTQ experiments
├── Compiled_LoRA.ipynb # Optimized vs base model speeds
├── prototype/ # initial mini experiments
└── README.md
```


## Getting Started
The notebooks are all self contained.

**Note:** You might not be able to view some of the Jupyter notebooks directly on the GitHub page due to GitHub’s rendering limitations. For the best experience, please download the notebooks and open them locally.

### Viewing the Project Locally

```
git clone https://github.com/ab4972/DeepChem-Model-Optimization.git
```

### WandB replication

```
pip install wandb
wandb login [your-api-key] # Get key from https://wandb.ai/authorize
```


## Key Features

1. **Parameter Efficiency**  
   Achieves _% parameter reduction with <_% accuracy drop using LoRA
2. **Quantization Aware Training**  
   Reduces model size by _x while maintaining _%+ accuracy
3. **Knowledge Distillation with LoRA:** 
    Distills knowledge from a large ChemBERTa teacher into a smaller, LoRA-adapted student
4. **Reproducible Workflow**  
   Full W&B integration for experiment tracking


## References
1. [ChemBERTa: Pre-trained Molecular Representation Model](https://arxiv.org/abs/2010.09885)  
2. [DeepChem](https://deepchem.io/)  

---

**Contributors** [Adam Banees](https://github.com/ab4972), [Shriya Kalakata](https://github.com/shriyakalakata), [Twinkle Gupta](https://github.com/twinklegupta013)

