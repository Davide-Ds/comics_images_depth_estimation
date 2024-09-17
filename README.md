# Depth Estimation in Comics Images

## Overview

This project focuses on solving the problem of **depth estimation** in comics images proposed in the [AI for Visual Arts Challenges (AI4VA) on Depth and Saliency](https://github.com/IVRL/AI4VA/tree/main). 

Comics present unique challenges for depth estimation, such as exaggerated perspectives and non-realistic visual elements, making it difficult to accurately infer depth from these stylized images. The goal of this project is to predict two types of depth values: **intra-depth** (depth within objects) and **inter-depth** (depth between objects).

We use deep learning models based on **Convolutional Neural Networks (CNNs)** to extract visual features from comics images and predict these depth values.

## Dataset

The **AI4VA dataset** is used in this project, consisting of comics images annotated with depth information. The dataset covers various comic styles and includes:
- **Depth Annotations**: Ground truth depth values for different image segments.
- **Comics Styles**: Depth annotations across different visual styles, requiring the model to generalize effectively.

You can download the dataset from: [Google Drive](https://drive.google.com/drive/folders/1C5ER7Trz7I-oyzV7YndNZZ6UJMuNTH10?usp=sharing).

## Installation and Setup

First, install the necessary libraries using the following command:

```bash
pip install requirements.txt
```

Ensure GPU support is available for faster model training.

## Project Structure

- **scripts/data_prepocessing.py**: module to load and preprocess the dataset. This includes normalizing images, resizing them to a uniform size, and loading depth annotations.
  

- **scripts/train_model.py**: Trains the model for depth estimation using CNN architectures. 

- **scripts/evaluate_model.py**: Evaluates the model on validation data and calculates the **mean squared error (MSE)** for intra-depth and inter-depth predictions. The results are saved to a text file with timestamps.

- **scripts/inference.py**: Runs inference on test images using the latest trained model and generates predictions, which are saved to a CSV file.

## Step-by-Step Guide

### 1. Setup Environment
   Install the required libraries:

   ```bash
   pip install requirements.txt
   ```

### 2. Download the Dataset
   Download the comics dataset from the provided Google Drive link and organize it in the following structure:
   
   ```plaintext
   data/
    ├── images/
       ├── train/
       ├── val/
       ├── test/
   annotations/
       ├── train-annotations.json
       ├── val-annotations.json
       ├── depth_TEST_segments.json
   ```
### 3. Data Exploration

Use the provided notebook "show_annotations_depth.ipynb" to explore the dataset and the annotations. Visualize a few images and their corresponding depth maps to understand the data.

### 4. Data Preprocessing
   Use the data_preprocessing.py script to load and preprocess images and annotations.

   ```bash
   python scripts/data_preprocessing.py
   ```

### 5. Train the Model
   Train the model by running the training script. The script will handle training and validation splits, early stopping, and model checkpointing.

   ```bash
   python scripts/train_model.py
   ```

   The training will save the best model based on validation performance.

### 6. Evaluate the Model
   Evaluate the trained model on validation data using the `evaluate_model.py` script. This script will load the model, run predictions on the dataset, and calculate the MSE.

   ```bash
   python scripts/evaluate_model.py
   ```

### 7. Inference on Test Data
   Perform inference on unseen test images by running the `inference.py` script. The predictions will be saved in CSV format.

   ```bash
   python scripts/inference.py
   ```

### 8. Results

The results of the evaluation for different models are stored in the `results/` directory. The **mean squared error (MSE)** values for intra-depth and inter-depth are used to assess model performance.

Example of evaluation results:

```
Evaluation written on: 2024-09-16 17:31
Model: /path/to/best_model.pth
MSE of Inter-depth: 1.536808777740302
MSE of Intra-depth: 2.438350742738203
Overall MSE: 1.9875797602392526
```

Example of inference results:

```
img_id category_id  pred_Intradepth  pred_Interdepth
0      131          28         0.341507         0.700975
1      158          22         0.223486         0.466395
2      163           1         0.372284         0.699417
3      188          16         0.344535         0.519607
4      198          12         0.426177         0.770635
5      207          25         0.392649         0.699364
6      213          27         0.475495         0.848142
7      245          24         0.337580         0.594709
8      261          23         0.398065         0.787483
9      269          11         0.316297         0.644748
10     275          28         0.390858         0.697132
11     282          12         0.418563         0.764017
12       1          22         0.451510         0.864075
13       8          23         0.729622         1.435773
14      21          16         0.417270         0.823564
15      25           1         0.271126         0.547276
16      38          19         0.654990         1.304213
17      62          13         0.796502         1.585559
18      68          25         0.604128         1.144789
19     101          24         0.445548         0.941498
```

### 9. Run the image_depth_estimation.ipynb
This [nootebook](https://github.com/Davide-Ds/comics_images_depth_estimation/blob/master/image_depth_estimation.ipynb) include all the pipeline's steps. Run it to have a detailed overview of the project.

## Future Work

Possible improvements include:
- Experimenting with **attention mechanisms** to better capture spatial relationships in comics images.
- Exploring **multi-task learning** by combining depth estimation with other visual tasks, such as segmentation.
