# Neutral_Network_ML: Predicting Funding Success with Deep Learning

## 📘 Overview

This is a deep learning project that leverages a neural network model to predict the success of funding applications for the fictional organization **Alphabet Soup**. The project demonstrates proficiency in data preprocessing, model development, evaluation, and optimization using Python and Keras.

---

## Project Highlights

- **Objective**: Predict the likelihood of a funding application being successful based on various input features.
- **Dataset**: Utilized a synthetic dataset representing funding applications, with a target variable indicating success.
- **Model Architecture**:
  - **Layers**: 3
  - **Neurons per Layer**: 32, 16, 1
  - **Activation Functions**: ReLU (hidden layers), Sigmoid (output layer)
- **Compilation & Training**:
  - **Optimizer**: Adam
  - **Loss Function**: Binary Crossentropy
  - **Metrics**: Accuracy
- **Evaluation**:
  - Achieved an accuracy of 72% on the test dataset.
  - Implemented a confusion matrix to assess model performance.

---

## 🧪 Key Steps in the Analysis

1. **Data Preprocessing**:
   - Target Variable: `IS_SUCCESSFUL`
   - Feature Variables: `STATUS`, `ASK_AMT`, `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`
   - Excluded Variables: `EIN`, `NAME`
   - Applied one-hot encoding to categorical variables.
   - Standardized numerical features.

2. **Model Development**:
   - Constructed a neural network with three layers.
   - Applied ReLU activation functions to hidden layers and Sigmoid to the output layer.
   - Compiled the model using the Adam optimizer and binary crossentropy loss function.

3. **Model Evaluation**:
   - Evaluated the model on a test dataset.
   - Generated a confusion matrix to visualize performance.

---

## Repository Structure

- `AlphabetSoupCharity_Optimization.ipynb`: Jupyter notebook containing the complete analysis, including data preprocessing, model development, training, and evaluation.

---

## Technologies Used

- Python
- Keras
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Outcome

The project successfully demonstrates the application of deep learning techniques to a real-world problem, showcasing skills in data preprocessing, model development, and evaluation. The achieved accuracy and confusion matrix provide insights into the model's performance and areas for potential improvement.

