# Multimodal-Persuasive-Technique-Detection

The website of the shared task, with the submission instructions, updates on the competition and the live leaderboard can be found here: [https://propaganda.math.unipd.it/semeval2024task4/](https://propaganda.math.unipd.it/semeval2024task4/)

__Table of contents:__

- [Competition](#semeval-2024-task4-corpus)
  - [List of Versions](#list-of-versions)
  - [Task Description](#task-description)
  - [Data Format](#data-format)
  - [Format checkers](#format-checkers)
  - [Scorers](#scorers)
  - [Baseline](#baseline)
  - [Licensing](#licensing)
  - [Citation](#citation)

## List of Versions
* __v0.1 [2023/09/21]__ - Training, validation and development sets memes released. Annotations for training and validation sets released.
* __v0.1 [2023/08/20]__ - Trial data released.


## Task Description

**Subtask 1:** Given only the “textual content” of a meme, identify which of the 20 persuasion techniques, organized in a hierarchy, it uses. If the ancestor node of a technique is selected, only a partial reward is given. This is a hierarchical multilabel classification problem. Details on the hierarchy will be provided in the corresponding section below. If you need additional annotated data to solve this task, you can check the [PTC corpus](https://propaganda.math.unipd.it/ptc/)" as well as the [SemEval 2023 task 3 data](https://propaganda.math.unipd.it/semeval2023task3/).

**Subtask 2a:** Given a meme, identify which of the [22 persuasion techniques](https://propaganda.math.unipd.it/semeval2024task4/definitions22.html), organized in a hierarchy, are used both in the textual and in the visual content of the meme (multimodal task). If the ancestor node of a technique is selected, only partial reward will be given. This is a hierarchical multilabel classification problem. Details on the hierarchy will be provided in the corresponding section below. 

**Subtask 2b:** Given a meme (both the textual and the visual content), identify whether it contains a persuasion technique (at least one of the [22 persuasion techniques](https://propaganda.math.unipd.it/semeval2024task4/definitions22.html) we considered in this task), or no technique. This is a binary classification problem. Note that this is a simplified version of subtask 2a in which the hierarchy is cut at the first two children of the root node. 

## Data Format

The datasets are JSON files. The text encoding is UTF-8.

### Input data format
#### Subtask 1:
An object of the JSON has the following format:
```
{
  id -> identifier of the example,
  text -> textual content of meme
  labels -> list of propaganda techniques appearing in the meme (based on hierarchy)
}
```
##### Example
```
{
        "id": "125",
        "labels": [
            "Reductio ad hitlerum",
            "Smears",
            "Loaded Language",
            "Name calling/Labeling"
        ],
        "text": "I HATE TRUMP\n\nMOST TERRORIST DO",
}
```
#### Subtask 2a:
An object of the JSON has the following format:
```
{
  id -> identifier of the example,
  text -> textual content of meme
  image -> name of the image file containing the meme
  labels -> list of propaganda techniques appearing in the meme (based on hierarchy)
}
```
##### Example
```
{
        "id": "125",
        "labels": [
            "Loaded Language",
            "Name calling/Labeling"
        ],
        "text": "I HATE TRUMP\n\nMOST TERRORIST DO",
        "image": "125_image.png"
}
```

#### Subtask 2b:
An object of the JSON has the following format:
```
{
  id -> identifier of the example,
  label -> ‘propandistic’ or ‘not-propagandistic’,
  text -> textual content of meme,
  image -> name of the image file containing the meme
}
```
##### Example
```
{
        "id": "1234",
        "label": ‘propandistic’,
        "text": "I HATE TRUMP\n\nMOST TERRORIST DO",
        "image" : "prop_meme_1234.png"
}
```
<!--![125_image](https://user-images.githubusercontent.com/33981376/99262849-1c62ba80-2827-11eb-99f2-ba52aa26236a.png)-->
<img src="https://user-images.githubusercontent.com/33981376/99262849-1c62ba80-2827-11eb-99f2-ba52aa26236a.png" width="350" height="350">

### Prediction Files Format

A prediction file, for example for the development set, must be one single JSON file for all memes. The entry for each meme must include the fields "id" and "labels". As an example, the input files described above would be also valid prediction files. 
We provide format checkers to automatically check the format of the submissions (see below). 

If you want to check the performance of your model on the development and test (when available) sets, upload your predictions' file to the website of the shared task: https://propaganda.math.unipd.it/semeval2024task4/. 
See instructions on the website about how to register and make a submission. 


## Baselines, Scorers and Hierarchy

### Subtask 1

This is a hierarchical multilabel classification task. 
For details on the function implemented, see [1] section 6. Moreover, we provided an intuition on the main page of the website. 
The list of valid output classes are in the file "output-classes-subtask1.txt" in the zip file coming together with this README. 
Assuming the gold label file is in "data/subtask1/validation.json", the following command evaluates subtask 1 on a baseline that always predicts smears:
```
python3 subtask_1_2a.py --gold_file_path data/subtask1/validation.json --pred_file_path subtask1_validation.json.txt
```
The output of the previous command is:
```
f1_h=0.36509	prec_h=0.45733	rec_h=0.30381
```

### Subtask 2a

The subtask is the same as Subtask 1, i.e. a hierarchical multilabel classification task [1] (section 6). 
The list of valid output classes are in the file "output-classes-subtask2a.txt" in the zip file coming together with this README. 
An example of invokation of the scorere on a baseline that always predicts smears, is the following
```
python3 subtask_1_2a.py --gold_file_path data/subtask2a/validation.json --pred_file_path subtask2a_validation.json.txt
```
The output of the previous command is:
```
f1_h=0.45885	prec_h=0.68200	rec_h=0.34572
```

### Subtask 2b
This is a binary classification task. 
The two classes, non_propagandistic and propagandistic, are in the file "binary_classes.txt"
An example of invokation of the scorere	on a baseline that always predicts propagandistic, is the following
```
python3 subtask_2b.py --gold_file_path data/subtask2b/val.json --pred_file_path val.json.pred.txt --classes_file_path binary_classes.txt 
```
The output of the previous command is:
```
macro-F1=0.25000	micro-F1=0.33333
```


## Licensing

The dataset is available on the [competition website](https://propaganda.math.unipd.it/semeval2024task4/). 
You'll have to accept an online agreement before downloading and using our data, which is strictly for research purposes only and cannot be redistributed or used for malicious purposes such as but not limited to manipulation, targeted harassment, hate speech, deception, and discrimination.

## Contact
You can contact us at the emails listed on the website. 


[1] https://www.svkir.com/papers/Kiritchenko-et-al-hierarchical-AI-2006.pdf
