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

To improve inferences, full fine-tuning approach was performed and the results evaluated with ROUGE Metrics. ROUGE Metrics may not perfect, but it does indicate the overall increase in summarization effectiveness that we have accomplished by fine-tuning. Furthermore, PEFT fine-tuning was performed and evaluation of results using the ROUGE metrics too. Comparing both results shows the benefit of PEFT outweighing the slightly-lower performance metrics.  .

## Getting Started
The following steps will help you get started:

This projects was originally carried out in an AWS Sagemaker notebook. The Instance type from SageMaker Comprises of Eight vCPU, 32 gigabyte. This is the AWS instance type from SageMaker, ml.m5.2xl


![instance_type](https://github.com/kennethugo/PEFT-LoRA_Fine-tuning/assets/50516854/ca0fa9b2-54ec-4737-a8a7-35c1e17bcceb)



1. Open the notebook in Goggle colab
https://colab.research.google.com/drive/1YRQlMInx7nfsf2qsBLDw9lx33a7FetnB#scrollTo=Zipuq1NHtqDq
3. Codes and comments are available to explain the different steps:

* Setting up Kernel, Load Required Dependencies, Dataset and LLM
* Importing packages
* Downloading dataset
* Creating and splitting the dataset 
* Calculating class weights
* Data Preprocessing and Augmentation 
* Metrics
* Model Building
* Model training
* Fine tuning
* Evaluation and validation tests
* Inference
* 1- Set up Kernel, Load Required Dependencies, Dataset and LLM
    * 1.1 - Set up Kernel and Required Dependencies
      1.2 - Load Dataset and LLM
      1.3 - Test the Model with Zero Shot Inferencing
      2 - Perform Full Fine-Tuning
* 2.1 - Preprocess the Dialog-Summary Dataset
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

The Road and Field Binary Classification Project involve the categorization of images into two classes: roads and fields. The dataset used for this project consists of a diverse collection of road and field images provided by Trimble / Bilberry. The dataset show 45 images of fields and 108 images of road. 

### Data Preprocessing

Prior to training the classification model, extensive data preprocessing was performed. First of all, in solving the problem of data imbalance, I considered techniques like oversampling and class weighting in addition to data augmentation. Due to the small number of the dataset, applying data augmentation techniques to artificially increase the size and diversity of the training dataset was inevitable. So I used common augmentations like random rotations, flips, zooms to help augment and improve diversity and model's robustness. The images were resized to a consistent resolution, maintaining same size of (244, 244) mobileNETv2 was preprocessed with. 

After oversampling the dataset using the Borderline SMOTE method, both classes had equal number of examples but going through the oversampled images, I realized the qualities were poor and there may be no new information for the model to learn and this can further increase over fitting. This left me with the option of class weighting. Class weighting will assign different weights to classes during training, giving higher weight to the minority class, which effectively increases its importance during gradient updates. So I used the class weighting technique by calculating class weight of the two classes and applied this during training. This reduced over fitting and bias towards the majority Class and also reduce poor generalization to the minority class.. The best solution to handle the data imbalance could have been to generate or collect more data to add to the minority field class, but It was not stated if this option was allowed in this exercise. 
Going forward, the dataset was split into training and validation sets using the tf.keras image_dataset_from_directory. As best practice, the same preprocessing that was used during training of mobilenetV2 was also applied.  This helps achieve desired result since transfer learning of the mobile will be used.


### Model Architecture

Transfer learning approach was adopted using the MobileNetV2 architecture, a pre-trained Convolutional neural network (CNN) model. I used MobilenetV2 for transfer learning because of its designed to provide fast and computationally efficient performance, even good for lightweight deep learning tasks. The pre-trained MobileNetV2 was fine-tuned for the binary classification task. To enhance the model's performance, additional layers were appended, including global average pooling, dropout layers were added, and a dense layer. During fine-tuning, more dropout values and dense layers with L2 regularization were experimented with to reduce overfitting and improve models performance on validation datasets.

### Training and Evaluation

The model was trained using a combination of the fine-tuned architecture and the Adam optimizer. A Binary Cross-Entropy loss function was employed, and class weights were utilized to address the class imbalance between roads and fields. The training process was monitored using metrics such as accuracy, precision, recall. Tensorboard was used to visualize the curves and graphs of the various metrics. class weighting was used to assign different weights to different classes during training.  This is to reduce bias toward the Majority class when predicting that majority class. This will also stop the tendency of the model not generalize well to the minority class.

The model's performance was evaluated on a validation dataset that was distinct from the training data. Early stopping was implemented to prevent overfitting, and the best weights were restored to ensure optimal generalization.

### Results

Considering the size of the overall dataset, the model achieved impressive results in classifying road and field images. Although overfitting was not completely removed, bias towards predicting the majority class (field class) was eliminated using class weighting. Poor generalization of the model to the minority class (the road class) was also reduced; there was no overfittiing to the majority class for a model like this.  Overfitting was generally reduced and controlled during fine-tuning using dropout and l2 regularization. The evaluation metrics showcased the model's effectiveness in distinguishing between the two classes. The precision, recall, and F1-score demonstrated the model's ability to provide reliable predictions.



## Conclusion

The Road and Field Binary Classification Project demonstrates the application of transfer learning and advanced computer vision techniques to the task of classifying road and field images. This exercise also provides solution to solve the problems that could be caused by small and imbalance datasets like overfitting and bias toward a majority Class, or poor generalization of the model to the minority class etc. By leveraging a pre-trained model, fine-tuning, and implementing data augmentation and class weighting impressive classification results were achieved. The project's methodology serves as a valuable foundation for future work or similar image classification tasks and can be extended to other domains. There are rooms for Improvement too.

For detailed code implementation and usage instructions, refer to the [**Getting Started**](##Getting-Started) section above.


## Further work
While this project has yielded promising results, there is ample room for future improvements and enhancements.


