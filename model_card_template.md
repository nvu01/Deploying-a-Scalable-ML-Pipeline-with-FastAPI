# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random Forest is used with default hyperparameters.

## Intended Use

The model should be used to predict whether an individual's income exceeds $50k/year based on census data.

## Training Data

Train data is 75% of the dataset.

## Evaluation Data

Evaluation data is the other 25% of the dataset.

## Metrics

Metrics used and their results after training and testing the model:  
Precision: 0.7598  
Recall: 0.6111  
F1: 0.6774

## Ethical Considerations

The model could potentially reinforce some societal bias because some demographic features can be correlated with race, gender, etc.

## Caveats and Recommendations

It is recommended to explore feature importance in further depth to understand which variables contribute most to the predictions.

The model’s performance should be periodically reassessed, and the model should be updated with new data to maintain its accuracy and relevance.