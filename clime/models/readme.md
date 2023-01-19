# Models
Models must be a class with methods from [clime.models.base_model](./base.py):

## __init__
trains the model
  - input: data

## predict
predicts y given X
  - input: X

## predict_proba
predicts p(X) given X
  - input: X

# Balancers
Balance a model that has already been trained

## adjust_boundary
Adjust decision boundary away from minority class to protect it

## adjust_proba
Increase probability of minority class predictions to protect it
