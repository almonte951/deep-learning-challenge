# Deep Learning Challenge
## Module 21 Challenge
### By Eduardo Almonte

The task involves creating a machine learning model for Alphabet Soup, a nonprofit organization, to predict the success of applicants who have received funding. The goal is to develop a binary classifier using neural networks to predict if funded organizations will be successful, based on a dataset containing historical data of over 34,000 organizations.

### Overview of the Analysis
The purpose of this analysis is to develop a binary classification model that can predict whether organizations funded by Alphabet Soup are likely to succeed. Using machine learning techniques, specifically neural networks, the goal is to create a model that classifies each organization's success based on the features available in the dataset. This will help Alphabet Soup prioritize funding for organizations with the highest likelihood of success.

### Results

### Data Preprocessing
### Target Variable:
IS_SUCCESSFUL (indicates whether the funded organization was successful).
### Features:
APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT.
### Removed Variables:
EIN and NAME were removed because they are identification columns and do not contribute to predicting success.

### Compiling, Training, and Evaluating the Model
### Neural Network Configuration:
Input Layer: Number of neurons equal to the number of input features.
### Hidden Layers:
Two hidden layers were used. The first hidden layer had 64 neurons, and the second hidden layer had 32 neurons.
Activation function: ReLU (Rectified Linear Unit) for both layers, which is a common choice for hidden layers as it helps with gradient-based optimization techniques.
### Output Layer:
One neuron with a sigmoid activation function to output a probability (0 to 1) for binary classification.
### Model Performance:
The initial model's accuracy was below the target of 75%.
### Steps Taken to Improve Performance:
Increased the number of neurons in the hidden layers to allow the model to learn more complex patterns.
Added more training epochs to give the model additional iterations to learn from the data.
Experimented with dropout regularization to avoid overfitting by randomly ignoring certain neurons during training.
Tuned the learning rate of the optimizer to improve convergence.

Summary of the Deep Learning Model Results
The deep learning model developed for Alphabet Soup successfully predicted the likelihood of an organization's funding success, achieving an accuracy close to or above the target of 75% after iterative optimizations. The process involved preprocessing the data, building a neural network with two hidden layers, and tuning hyperparameters such as the number of neurons, epochs, and dropout regularization to improve performance.
