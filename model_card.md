# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This Model was Developed by Benjamin Kenner during a udacity Nanodegree Course in June 2022
It is a Random Forest Model with default Parameters, provided by the sklearn Python package.
The Model was trained and validated with a simple train test split of 80/20.

## Intended Use

This model can be used to predict the salary of an individual, based on the following features:
* "age"
* "workclass"
* "fnlgt"
* "education"
* "education-num"
* "marital-status"
* "occupation"
* "relationship"
* "race"
* "sex"
* "capital-gain"
* "capital-loss"
* "hours-per-week"
* "native-country"

## Training Data

The Training data was provided by the udacity course and samples 32561 examples.
For training in the validation step 26048 of those examples where used. For training for the final model, all examples where used.

## Evaluation Data

For evaluation 6513 examples where used.

## Metrics

To meassure the models performance three different metrics where used.

* precision = tp / (tp + fp)
* recall = tp / (tp + fn)
* fbeat(beta=1) = ((1+beta*beta)*(precision*recall))/(beta*beta*precision+recall)

## Ethical Considerations

The data used for this model includes sensitive information about individuals and might be biased based on those features. Therefor the predictions made by this model have to be used carefully.

## Caveats and Recommendations

The training of this model was done for demonstrative reasons. There was no further parameter optimization or feature engineering to optimize the results.