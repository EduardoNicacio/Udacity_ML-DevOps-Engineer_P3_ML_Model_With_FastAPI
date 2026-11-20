import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return float(precision), float(recall), float(fbeta)


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(
    model, encoder, lb, data, feature, cat_features, label="salary"
):
    """Compute and print model metrics for every value of a feature.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained encoder used during inference.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained label binarizer used during inference.
    data : pd.DataFrame
        Labeled data to slice, including the label column.
    feature : str
        Name of the categorical feature to slice on.
    cat_features : list[str]
        Names of the categorical features used during training.
    label : str
        Name of the label column in `data` (default="salary").

    Returns
    -------
    None
        Metrics are printed to stdout, one line per unique feature value.
    """
    for value in data[feature].unique():
        slice_data = data[data[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            slice_data,
            categorical_features=cat_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        print(
            f"Feature: {feature}, Value: {value} \u2192 "
            f"precision: {precision:.2f}, recall: {recall:.2f}, "
            f"f1: {fbeta:.2f}"
        )


def save_model(model, path):
    """Save a model or preprocessing artifact with joblib.

    Inputs
    ------
    model : object
        Fitted model or preprocessing object to persist.
    path : str
        Destination path. Parent directories are created if needed.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Load a model or preprocessing artifact saved with joblib.

    Inputs
    ------
    path : str
        Path to the joblib file.

    Returns
    -------
    object
        The deserialized model or preprocessing object.
    """
    return joblib.load(path)
