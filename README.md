# Detecting Persuasion Techniques in Memes

## Project Description

"Detecting Persuasion Techniques in Memes" is a research project aimed at identifying and analyzing various persuasion techniques used in memes. This project is part of the SemEval2024
shared task on "Multilingual Detection of Persuasion Techniques in Memes." The challenge involves understanding and classifying the persuasive elements in memes, which may include logical 
fallacies and emotional appeals, among others.

Memes are a potent medium in online discourse, often used in disinformation campaigns. By combining text and images, they can effectively influence public opinion. 
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

```bash
mkdir -p TextFeatures/subtask1a/text-embeddings-3-large
mkdir -p TextFeatures/subtask1a/text-embeddings-3-small
mkdir -p TextFeatures/subtask1a/mBERT
mkdir -p TextFeatures/subtask1a/multilingual-ner
```

Create the same structure for subtask2a under `TextFeatures` directory. 

### Step 3: Setting Up Feature Storage for Image Features
Additionally, set up directories under ImageFeatures to store image features extracted for Subtask 2a. Use the following commands to create these directories:
```bash
mkdir -p ImageFeatures/CLIP-ViT
mkdir -p ImageFeatures/ResNet
```

### Step 4: Extracting Text and NER Features and Image Features

Use the provided Jupyter notebooks to extract and save the features. Run the following notebooks:
- **Feature Extractor.ipynb**: This notebook extracts general text features using models such as `text-embeddings-3-large` and `text-embeddings-3-small` from OpenAI, and `bert-base-multilingual-uncased`.
- **Extract NER Embeddings.ipynb**: This notebook is specifically used for extracting Named Entity Recognition (NER) features using the `Babelscape/wikineural-multilingual-ner` model.
- **Feature Extractor.ipynb**: This notebook has code to extract visual features for subtask2a using CLIP model. 

Then, open each notebook and execute the cells according to the instructions provided within them.

### Step 5: Reusing Extracted Features

Once the features are extracted and stored in the `TextFeatures` directory, you can proceed with training your models. The training scripts are configured to automatically load these pre-extracted features, significantly speeding up the training process.

## Obtaining Milestone 2 Results

To reproduce the results for Milestone 2 using our baseline models, follow the instructions below. These results pertain to the initial evaluations conducted with our baseline models for both text-only and multimodal approaches.

### Running Baseline Models

To obtain the results for the baseline models, you need to run the following Jupyter notebooks:
- **Baseline softmax.ipynb**: This notebook executes our softmax baseline model which focuses on the text-only approach for Subtask 1.
- **BaseLine subtask 2a.ipynb**: This notebook is used for the multimodal approach combining both text and image data, specifically for Subtask 2a.

## Obtaining Milestone 3 Results

Milestone 3 focuses on refining the performance of our models through hyperparameter tuning and ensemble methods. This stage is critical for enhancing the accuracy and robustness of our predictions, especially for multimodal data that includes both text and image content.

### Hyperparameter Tuning

To begin hyperparameter tuning, run the notebooks that start with `FineTune`. These notebooks contain code for fine-tuning various models using the features extracted previously for both text and image content.

### Best Hyperparameters for Ensemble Models

In our ensemble, each model was fine-tuned with specific hyperparameters that optimized its performance. Below, we detail the best hyperparameters used for each model included in the ensemble.

#### Model 1: BERT-Base-Multilingual-Uncased (subtask1a)

- **alpha:** 0.8551238693613787
- **batch_size:** 256
- **beta:** 0.9636246518811256
- **beta1:** 0.8795857325304084
- **learning_rate:** 0.000328898551319524
- **optimizer:** "adam"

#### Model 2: Text-Embeddings-3-Small (subtask1a)

- **alpha:** 0.6903448674283283
- **batch_size:** 256
- **beta:** 0.5953036715609454
- **beta1:** 0.8889795457914553
- **learning_rate:** 0.00008545944172126079
- **threshold:** 0.7579501306367982
- **optimizer:** "adam"

#### Model 3: Text-Embeddings-3-Large (subtask1a)

- **alpha:** 0.8755203100909268
- **batch_size:** 256
- **beta:** 0.6818375897649283
- **beta1:** 0.8040811213879919
- **learning_rate:** 0.00005924267168802313
- **threshold:** 0.6965199717281236
- **optimizer:** "adam"

#### Model 4: Multilingual-NER concatenated with Text-Embeddings-3-Large (subtask1a) [FineTuning-openAI-NER-models.ipynb]

- **alpha:** 0.9654478666765854
- **batch_size:** 128
- **beta:** 0.7870273949525248
- **beta1:** 0.9476979307645412
- **learning_rate:** 0.00002919434743481965
- **threshold:** 0.70148548183942
- **optimizer:** "adam"

#### Model 5: dslim/bert-large-NER concatenated with Text-Embeddings-3-Large (subtask1a)

- **alpha:** 0.7899005629013283
- **batch_size:** 256
- **beta:** 0.6740115397343491
- **beta1:** 0.8278023344142735
- **learning_rate:** 0.00006878876239114434
- **threshold:** 0.8328292373348003
- **optimizer:** "adam"

#### Model 6: MultiModal (Text-Embeddings-3-Small + CLIP ViT) (subtask 2a)
- **alpha:** 0.9596038456941688
- **batch_size:** 256
- **beta:** 0.8280858527581143
- **beta1:** 0.8866946466052821
- **learning_rate:** 0.00002027017555166722
- **threshold:** 0.777320492008404
- **optimizer:** "adam"

#### Model 7: MultiModal (Text-Embeddings-3-Large + CLIP ViT) (subtask 2a)
- **alpha:** 
- **batch_size:** 
- **beta:** 
- **beta1:** 
- **learning_rate:** 
- **threshold:** 
- **optimizer:** "adam"

#### Model 8: MultiModal (Text-Embeddings-3-Large with multilingual NER Features + CLIP ViT) (subtask 2a)
- **alpha:** 
- **batch_size:** 
- **beta:** 
- **beta1:** 
- **learning_rate:** 
- **threshold:** 
- **optimizer:** "adam"


These parameters were identified through a rigorous process of hyperparameter tuning, involving bayes search and validation on a held-out dataset to ensure that each model performs optimally within the ensemble.

#### Using Hyperparameters in Model Training

To utilize these hyperparameters in training, refer to the specific sections in the fine-tuning notebooks where these values are set before starting the training process. Ensure to adjust the paths and model configurations as per your setup.

## Post-Training Setup and Evaluation

After training the models as per the hyperparameters discussed, the next steps involve organizing the predictions and evaluating the ensemble models for both Subtask 1a and Subtask 2a.

### Step 1: Create Directories for Storing Predictions

Before moving forward with predictions, set up directories to organize the output. Use the following commands to create the necessary directories:

```bash
mkdir -p Predictions
mkdir -p Predictions/subtask2a
```
### Step 2: Evaluate Ensemble Models
Once the models are trained, proceed to evaluate them using the ensemble approach. We have dedicated notebooks for these evaluations:

- Ensemble Evaluation Subtask 1a.ipynb
- Ensemble Evaluation Subtask 2a.ipynb
















