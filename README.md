# PEFT-LoRA_Fine-tuning of Flan-T5 model
This repository shows the step-by-step process of Fine-tuning using PEFT (Parameter Efficient Fine-Tuning) with LoRA for improving the Dialogue summarization capacity of the Flan-T5 model. 

# Table of Content
* [**Introduction**](##Introduction)
* [**Getting Started**](##Getting-Started)
* [**Methodology and Results**](##Methodology-and-Results)
* [**Conclusion** ](##Conclusion)
* [**Further work** ](##Further-work)

## Introduction
This project details a step-by-step process for full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) with prompt instructions. Additionally, it includes further fine-tuning of the Flan-T5 model with customized specific prompts for a custom-specific summarization task.

Imagine a scenario where, despite performing in-context learning with zero-shot, one-shot, or even few-shot techniques, the Language Model (LLM) performance does not meet your specific task requirements. In such cases, full fine-tuning could be a potential solution, but do you have the necessary compute resources? If yes, that's fantastic! However, also consider multiple-task fine-tuning to address the challenge of Catastrophic Forgetting. Alternatively, a computationally cost-effective approach to consider is PEFT. PEFT preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters, thereby utilizing fewer storage and compute resources. While there may be a slight trade-off in performance, considering the reduction in compute resources required, it could be a worthwhile consideration

The base model used in this project is the FLAN-T5 model. The dataset is the DialogSum dataset from HuggingFace.
