import pytest
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import performance_on_categorical_slice, load_model

@pytest.fixture
def setup():
    """
    Fixture to load the data and model components.
    """
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    df = pd.read_csv(data_path)
    train, test = train_test_split(df, random_state=42)
    
    cat_features = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

    # load the model, encoder, lb
    model = load_model(os.path.join(project_path, "model", "model.pkl"))
    encoder = load_model(os.path.join(project_path, "model", "encoder.pkl"))
    lb = load_model(os.path.join(project_path, "model", "lb.pkl"))

    return df, train, test, cat_features, model, encoder, lb

def test_split_size(setup):
    """
    Test if the sizes of the train and test datasets are as expected.
    The train dataset should be 75% of the original dataset.
    The test dataset should be 25% of the original dataset.
    """
    df, train, test, _, _, _, _ = setup 
    expected_train_size = int(len(df) * 0.75)
    expected_test_size = len(df) - expected_train_size
    assert len(train) == expected_train_size, f"Train size is {len(train)} instead of {expected_train_size}"
    assert len(test) == expected_test_size, f"Test size is {len(test)} instead of {expected_test_size}"


def test_binary_label(setup):
    """
    Test if the process_data function returns binary label
    """
    df, _, _, cat_features, _, _, _ = setup
    _,labels,_,_ = process_data(
        X=df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    assert all(map(lambda x: x==0 or x==1, labels)), "The label column contains values other than 0 or 1."


def test_performance_on_categorical_slice(setup):
    """
    Test if the results of performance_on_categorical_slice function are in the range 0 to 1.
    """
    df, _, test, cat_features, model, encoder, lb = setup
    # compute the performance on model slices
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            # calculate metrics
            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model
            )
            assert 0 <= p <= 1, f"The model's precision metric for {slicevalue} in {col} column is not between 0 and 1."
            assert 0 <= r <= 1, f"The model's recall metric for {slicevalue} in {col} column is not between 0 and 1."
            assert 0 <= fb <= 1, f"The model's f1 metric for {slicevalue} in {col} column is not between 0 and 1."
