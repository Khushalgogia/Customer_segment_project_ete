
# Customer Segmentation and Purchase Prediction

## Overview

This project focuses on leveraging data science techniques to predict customer purchase behavior and segment customers into Premium and Basic categories. The goal is to optimize marketing strategies and enhance customer engagement.

## Project Structure

### 1. Data Preprocessing Pipeline

- **Data Cleaning:**
  - Handled missing values and outliers to ensure data integrity.
  - Performed necessary transformations for consistent data format.

- **Feature Engineering:**
  - Extracted relevant features to enhance model performance.
  - Applied scaling and normalization for better convergence.

### 2. Exploratory Data Analysis (EDA)

- Conducted thorough exploratory data analysis to gain insights into customer behavior.
- Visualized key patterns and trends to inform decision-making.

### 3. Model Building Pipeline

- **Purchase Prediction Model:**
  - Utilized machine learning algorithms like RandomForest, BoostingClassifier, LogisticRegression, SVM and so on to predict customer purchase likelihood.
  - Implemented a robust pipeline for preprocessing, model training, and evaluation.

- **Customer Segmentation Model:**
  - Employed clustering techniques such kmeans for customer segmentation.
  - Developed a pipeline for seamless application of the segmentation model.

### 4. Results and Insights

- Achieved more than 70% of accuracy in predicting purchases.
- Identified distinctive characteristics of Premium and Basic customers through segmentation.

## How to Use

1. **Data Preprocessing:**
   - Execute the `preprocessing_pipeline.py` script to preprocess the data.

2. **Exploratory Data Analysis:**
   - Refer to the `EDA.ipynb` notebook for in-depth analysis and visualizations.

3. **Model Building:**
   - Use the `purchase_prediction_model.py` script for training and evaluating the purchase prediction model.
   - Execute the `customer_segmentation_model.py` script for customer segmentation.

## Dependencies

- Python 3.x
- Libraries: (all mentioned in requirements.txt)

## Usage Guidelines

- Clone the repository to your local machine.
- Ensure all dependencies are installed using `requirements.txt`.
- Follow the provided scripts and notebooks for step-by-step execution.


Feel free to explore the codebase and adapt it for your specific needs. Feedback and contributions are highly welcomed!





