
## Overview of the Analysis
This analysis helps to develop a deep learning model to predict the success of funding applications for Alphabet Soup. 

## Results

### Data Preprocessing
- **Target Variable:**  "IS_SUCCESSFUL"
- Feature Variables (Model Inputs):

STATUS,
ASK_AMT,
APPLICATION_TYPE,
AFFILIATION,
CLASSIFICATION,
USE_CASE,
ORGANIZATION,
INCOME_AMT,
SPECIAL_CONSIDERATIONS.
- Removed Variables (Not used in training): EIN, NAME


### Compiling, Training, and Evaluating the Model
- **Neural Network Structure:**
  - Number of Layers: 3
  - Neurons per Layer: 32, 16, 1
  - Activation Functions: ReLU, Sigmoid
- **Model Performance:**
  - Accuracy: 72.8%
  - Loss: 0.55
- **Optimization Steps Taken:**
  - Adjusted the number of neurons and layers
  - Tuned learning rate and batch size
  - Applied dropout to reduce overfitting

## Summary
The deep learning model achieved moderate accuracy but could be further optimized.

