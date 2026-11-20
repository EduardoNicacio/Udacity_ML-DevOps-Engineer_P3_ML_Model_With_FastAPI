# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a random forest classifier built with scikit-learn
(`RandomForestClassifier`) using 100 trees, no maximum depth, `random_state=42`
and `n_jobs=-1`. It predicts whether a Census respondent's annual income is
greater than 50K USD. The positive class (`>50K`) is encoded as 1 and the
negative class (`<=50K`) is encoded as 0.

## Intended Use

The model is intended as a demonstration of an end-to-end machine learning
DevOps workflow: data cleaning, model training, unit testing, a FastAPI
inference service, continuous integration with GitHub Actions, and continuous
deployment with Render. It is suitable for educational and research purposes
only. It must not be used to make decisions about real people, such as credit,
employment or benefits eligibility.

## Training Data

The model was trained on the Census Bureau income dataset provided with the
project (`data/census.csv`, 32,561 rows). A cleaned copy was produced by
stripping leading and trailing whitespace from every column name and string
value, and saved as `data/census_clean.csv` without renaming any column. The
salary column is the target and all other columns are used as features. No
rows were dropped.

## Evaluation Data

An 80/20 train/test split was used with `random_state=42` and stratification on
the salary column, giving 26,048 training rows and 6,513 test rows. The test
set was processed with the encoder and label binarizer fitted on the training
set, and was never seen by the model during training.

## Metrics

The model is evaluated with precision, recall and F1 (the F-beta score with
beta equal to 1). On the held-out test set the model achieved a precision of
0.7353, a recall of 0.6378 and an F1 score of 0.6831. Per-slice metrics for
every value of the education feature are written to `slice_output.txt` by the
training script.

## Ethical Considerations

The dataset contains sensitive demographic attributes such as sex, race,
marital status and native country, and income predictions can reflect and
amplify historical inequalities present in the data. Sliced evaluation shows
that performance varies across education levels and, by extension, likely
across other protected groups. Because the model outputs a binary income
bracket, it should never be used to make decisions about individuals, and any
real-world use would require a thorough fairness audit and human oversight.

## Caveats and Recommendations

The model is trained on historical Census survey data and reflects the
conditions and biases of the period in which the data was collected. The single
random train/test split provides a point estimate of performance rather than a
full confidence interval, and the positive class has substantially lower recall
than precision, so many high-income respondents are misclassified. Before any
serious use, the model should be retrained with k-fold cross-validation,
hyperparameter tuning, probability calibration and a systematic fairness
analysis across all demographic slices.
