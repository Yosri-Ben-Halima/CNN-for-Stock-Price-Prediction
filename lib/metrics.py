import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def percentage_error(predictions_df):
    """
    Calculates the Percentage Error (PE) for the predictions.

    Parameters
    ----------
    predictions_df : pandas.DataFrame
        A DataFrame containing at least two columns:
        - 'Actual': The actual target values.
        - 'Predicted': The predicted values from the model.

    Returns
    -------
    None
        The function modifies the input DataFrame in place by adding a new column:
        - 'PE': The Percentage Error for each prediction, calculated as 100 * (Actual - Predicted) / Actual.

    Description
    -----------
    This function computes the Mean Percentage Error (MPE) for each entry in the DataFrame. The MPE is calculated by comparing the actual values with the
    predicted values and is expressed as a percentage. The result is stored in a new column 'MPE' within the input DataFrame.

    Note
    ----
    The input DataFrame is modified in place, and the function does not return any value.
    """

    predictions_df['PE'] = 100*(predictions_df['Actual']-predictions_df['Predicted'])/predictions_df['Actual']


def binary_evaluation_metrics(df):
    """
    Calculate the Confusion Matrix, F1 Score, and Accuracy Score for binary price moves.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing at least two columns:
        - 'Binary Actual Change': Binary indicators of actual price changes (1 for up, -1 for down).
        - 'Binary Predicted Change': Binary indicators of predicted price changes (1 for up, -1 for down).

    Returns
    -------
    tuple
        A tuple containing:
        - pandas.DataFrame: The confusion matrix as a DataFrame, with rows and columns indicating the actual and predicted price moves.
        - float: The F1 Score for the binary classification.
        - float: The Accuracy Score for the binary classification.

    Description
    -----------
    This function computes evaluation metrics for binary price moves. It generates
    a confusion matrix to compare actual and predicted price movements. Additionally,
    it calculates the F1 Score and Accuracy Score to assess the performance of the
    model in classifying price changes as either up or down.

    Note
    ----
    The confusion matrix is returned as a DataFrame with labels `Actual Up`, `Actual Down`
    for the rows and `Predicted Up`, `Predicted Down` for the columns.
    """

    cm = confusion_matrix(df['Binary Actual Change'][1:], df['Binary Predicted Change'][1:], labels=[1, -1])
    cm_df = pd.DataFrame(cm, index=['Actual Up', 'Actual Down'], columns=['Predicted Up', 'Predicted Down'])
    f1 = f1_score(df['Binary Actual Change'][1:], df['Binary Predicted Change'][1:], labels=[1, -1])
    accuracy = accuracy_score(df['Binary Actual Change'][1:], df['Binary Predicted Change'][1:])

    return cm_df, f1, accuracy

