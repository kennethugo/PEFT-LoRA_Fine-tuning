# PEFT-LoRA_Fine-tuning of Flan-T5 model
This repository provides a detailed guide on fine-tuning the Flan-T5 model from HuggingFace using Parameter Efficient Fine-Tuning (PEFT) with LoRA to get an improved Dialogue summarization capacity of the new model. 

# Table of Content
* [**Introduction**](##Introduction)
* [**Getting Started**](##Getting-Started)
* [**Methodology and Results**](##Methodology-and-Results)
* [**Conclusion** ](##Conclusion)
* [**Further work** ](##Further-work)

## Introduction
This project details a step-by-step process for full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LoRA) prompt instructions (which is NOT THE SAME as prompt engineering!). Additionally, it includes further fine-tuning of the Flan-T5 model with customized specific prompts for a custom-specific summarization task.

Imagine a scenario where, despite performing in-context learning with zero-shot, one-shot, or even few-shot techniques, the Language Model (LLM) performance does not meet your specific task requirements. In such cases, full fine-tuning could be a potential solution, but do you have the necessary compute resources? If yes, that's fantastic! However, also consider multiple-task fine-tuning to address the challenge of Catastrophic Forgetting. Alternatively, a computationally cost-effective approach to consider is PEFT. PEFT preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters, thereby utilizing fewer storage and compute resources. While there may be a slight trade-off in performance, considering the reduction in compute resources required, it could be a worthwhile consideration. After fine-tuning for a specific task, use case, or tenant with LoRA, the result is that the original LLM remains unchanged and a newly-trained “LoRA adapter” emerges. This LoRA adapter is much, much smaller than the original LLM - on the order of a single-digit % of the original LLM size (MBs vs GBs).

The base model used in this project is the FLAN-T5 model. The FLAN-T5 model provides a high quality instruction model and can summarize text out of the box. The dataset is the DialogSum dataset from HuggingFace.

## Getting Started
The following steps will help you get started:

This projects was originally carried out in an AWS Sagemaker notebook. The Instance type from SageMaker Comprises of Eight vCPU, 32 gigabyte. This is the AWS instance type from SageMaker, ml.m5.2xl


![instance_type](https://github.com/kennethugo/PEFT-LoRA_Fine-tuning/assets/50516854/ca0fa9b2-54ec-4737-a8a7-35c1e17bcceb)



1. Open the notebook in Goggle colab
https://colab.research.google.com/drive/1YRQlMInx7nfsf2qsBLDw9lx33a7FetnB#scrollTo=Zipuq1NHtqDq
3. Codes and comments are available to explain the different steps:

* 1- Set up Kernel, Load Required Dependencies, Dataset and LLM
  
       1.1 - Set up Kernel and Required Dependencies
  
       1.2 - Load Dataset and LLM

       1.3 - Test the Model with Zero Shot Inferencing

* 2 - Perform Full Fine-Tuning
  
       2.1 - Preprocess the Dialog-Summary Dataset

       2.2 - Fine-Tune the Model with the Preprocessed Dataset

       2.3 - Evaluate the Model Qualitatively (Human Evaluation)

       2.4 - Evaluate the Model Quantitatively (with ROUGE Metric)
  
* 3 - Perform Parameter Efficient Fine-Tuning (PEFT)
      3.1 - Setup the PEFT/LoRA model for Fine-Tuning
  
      3.2 - Train PEFT Adapter
  
      3.3 - Evaluate the Model Qualitatively (Human Evaluation)
  
      3.4 - Evaluate the Model Quantitatively (with ROUGE Metric)

## Methodology and Results
### Dataset

The dataset used for this project is the [DialogSum Dataset](https://huggingface.co/datasets/knkarthick/dialogsum) from HuggingFace. DialogSum is a large-scale dialogue summarization dataset, consisting of 13,460 (Plus 100 holdout data for topic generation) dialogues with corresponding manually labeled summaries and topics. The language is english. DialogSum is a large-scale dialogue summarization dataset, consisting of 13,460 dialogues (+1000 tests) split into train (12460), testb(1500) and validation (500). Data field include; 
*dialogue: text of dialogue.
*summary: human written summary of the dialogue.
*topic: human written topic/one liner of the dialogue.
*id: unique file id of an example.


### Data Preprocessing

Prior to full fine-tuning the dataset was converted to the dialog-summary (prompt-response) pairs into explicit instructions for the LLM. An instruction was Prepended to the start of the dialog with Summarize the following conversation and to the start of the summary with Summary as follows:

Training prompt (dialogue):

Summarize the following conversation.

    Speaker 1: This is his part of the conversation.
    speaker 2: This is her part of the conversation.
    
Summary: 
Training response (summary):

Both Speaker 1 and Speaker 2 participated in the conversation.
Then preprocess the prompt-response dataset into tokens and pull out their input_ids (1 per token)

### Training and Evaluation - Full Fine tuning and PEFT LoRA fine-tuning
The model was trained using the built-in Hugging Face Trainer class (see the documentation [here](https://huggingface.co/docs/transformers/main_classes/trainer)) and passing the preprocessed train dataset with reference to the original model. Other training parameters were found experimentally

The model's performance was evaluated on a validation dataset that was distinct from the training data.The ROUGE-1 metrics was used. To improve inferences, full fine-tuning approach was performed and the results evaluated with ROUGE Metrics. ROUGE Metrics may not perfect, but it does indicate the overall increase in summarization effectiveness that we have accomplished by fine-tuning. 

Furthermore, PEFT fine-tuning was performed and evaluation of results using the ROUGE metrics too. PEFT/LoRA model was set up with a new layer/parameter adapter for fine-tuning. Using PEFT/LoRA, you are freezing the underlying LLM and only training the adapter.

### Results


Performing inference for the sample of the test dataset, with the original model, the  fully fine-tuned model and the PEFT-model, shows huge improvement of PEFT over the original model though not better than the fully fine tuned model. However the training of the PEFT model requires much less computing and memory resources (often just a single GPU). Comparing both results using ROUGE-1 shows the benefit of PEFT outweighing the slightly-lower performance metrics. 


## Conclusion

With PEFT-LoRA fine tuning less compute cost resources can be used to perform specific task fine tuning which in most of the times out performs a base model 

For detailed code implementation and usage instructions, refer to the [**Getting Started**](##Getting-Started) section above.


## Further work
While this project has yielded promising results, there is ample room for future improvements and enhancements.


