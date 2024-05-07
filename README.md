# Detecting Persuasion Techniques in Memes

## Project Description

"Detecting Persuasion Techniques in Memes" is a research project aimed at identifying and analyzing various persuasion techniques used in memes. This project is part of the SemEval2024
shared task on "Multilingual Detection of Persuasion Techniques in Memes." The challenge involves understanding and classifying the persuasive elements in memes, which may include logical 
fallacies and emotional appeals, among others.

Memes are a potent medium in online discourse, often used in disinformation campaigns. By combining text and imagery, they can effectively influence public opinion. 
This project seeks to detect these techniques through hierarchical multilabel classification tasks, analyzing both the textual and visual components of memes. 

The tasks are divided into:
- **Subtask 1:** Identification of persuasion techniques from the textual content alone.
- **Subtask 2a:** Multimodal analysis, identifying persuasion techniques from both text and image.

This repository contains the code, models developed for the SemEval2024 competition, providing tools to tackle the challenges of detecting persuasion in multimodal content.

## Data Access and Prediction Submission

To access the data for this project and submit your predictions, register at the following link:

[SemEval2024 Registration](https://propaganda.math.unipd.it/semeval2024task4/)

After registration, you will be able to download the data and participate in the tasks by submitting your predictions through the provided platform.

## Our Approach

In our project "Detecting Persuasion Techniques in Memes", we address the hierarchical detection of persuasion techniques using a novel approach that considers a top-down hierarchical classification. This method provides the neural classifiers with more context at each level of the hierarchy, facilitating more accurate and context-aware predictions.

### Hierarchical Classification

We utilize a hierarchical classification strategy that begins at the root level and progresses deeper based on the predictions at each parent level. This approach not only respects the inherent hierarchy within the persuasion techniques but also enhances the contextual understanding necessary for accurate classification.

### Custom Loss Function

To effectively train our model within this hierarchical framework, we developed a custom loss function tailored for this task. This loss function is designed to manage the relationships between parent and child nodes in the hierarchy, penalizing discrepancies and improving the model’s ability to make consistent predictions across levels.

## Feature Extraction Process

To ensure efficient training and reduce computational overhead during each epoch, our project employs a pre-processing step where we extract features from both text and image data. These features are saved in a pickle file for reuse throughout the training process.

### Step 1: Data Preparation

After registering and downloading the necessary datasets from the [task webpage](https://propaganda.math.unipd.it/semeval2024task4/), place the data in the root directory of this repository.

### Step 2: Setting Up Feature Storage

Create a directory named `TextFeatures` in the root directory of the repository. Within this directory, create subfolders for each model used to extract features:

- mkdir -p TextFeatures/subtask1a/text-embeddings-3-large
- mkdir -p TextFeatures/subtask1a/text-embeddings-3-small
- mkdir -p TextFeatures/subtask1a/mBERT
- mkdir -p TextFeatures/subtask1a/multilingual-ner

Create the same structure for subtask2a under `TextFeatures` directory. 


### Step 3: Extracting Text and NER Features

Use the provided Jupyter notebooks to extract and save the features. Run the following notebooks:
- **Feature Extractor.ipynb**: This notebook extracts general text features using models such as `text-embeddings-3-large` and `text-embeddings-3-small` from OpenAI, and `bert-base-multilingual-uncased`.
- **Extract NER Embeddings.ipynb**: This notebook is specifically used for extracting Named Entity Recognition (NER) features using the `Babelscape/wikineural-multilingual-ner` model.
- **Feature Extractor.ipynb**: This notebook has code to extract visual features for subtask2a using CLIP model. 

Then, open each notebook and execute the cells according to the instructions provided within them.

### Step 4: Reusing Extracted Features

Once the features are extracted and stored in the `TextFeatures` directory, you can proceed with training your models. The training scripts are configured to automatically load these pre-extracted features, significantly speeding up the training process.

## Obtaining Milestone 2 Results

To reproduce the results for Milestone 2 using our baseline models, follow the instructions below. These results pertain to the initial evaluations conducted with our baseline models for both text-only and multimodal approaches.

### Running Baseline Models

To obtain the results for the baseline models, you need to run the following Jupyter notebooks:
- **Baseline softmax.ipynb**: This notebook executes our softmax baseline model which focuses on the text-only approach for Subtask 1.
- **BaseLine subtask 2a.ipynb**: This notebook is used for the multimodal approach combining both text and image data, specifically for Subtask 2a.








