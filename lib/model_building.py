from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import pandas as pd


def build_model(input_shape):
    """
    Builds and compiles a Convolutional Neural Network (CNN) model.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data, typically (window_size, 1, 1) where
        window_size is the number of time steps used in the model.

    Returns
    -------
    model : keras.Sequential
        A compiled CNN model ready for training.

    Description
    -----------
    This function constructs a CNN model using Keras' Sequential API. The
    model includes two convolutional layers followed by max pooling, a
    flattening layer, and two dense layers. The final dense layer outputs
    a single value, which is suitable for regression tasks such as stock
    price prediction. The model is compiled with the Adam optimizer and
    mean squared error as the loss function.
    """
    
    model = Sequential([
        Conv2D(32, (3, 1), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 1)),
        Conv2D(64, (3, 1), activation='relu'),
        MaxPooling2D((2, 1)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

def get_callbacks():
    """
    Returns a list of callbacks for training the model.

    Returns
    -------
    list
        A list containing Keras callbacks: EarlyStopping and ReduceLROnPlateau.

    Description
    -----------
    This function creates and returns two Keras callbacks to be used during 
    model training:
    
    - `EarlyStopping`: Monitors the validation loss and stops the training
      if the loss does not improve for a specified number of epochs (`patience`).
      Additionally, it restores the best model weights encountered during training.
    
    - `ReduceLROnPlateau`: Monitors the validation loss and reduces the learning
      rate by a specified factor if the loss does not improve for a certain number
      of epochs (`patience`). The learning rate is reduced until it reaches a 
      specified minimum value (`min_lr`).

    These callbacks help to prevent overfitting and ensure efficient training.
    """

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )

    return [early_stopping, reduce_lr]


def run_predictions(model, X_test, y_test, dates):    
    """
    Generates model predictions on the test set and prepares a DataFrame with
    actual and predicted changes.

    Parameters
    ----------
    model : keras.Model
        The trained Keras model used for making predictions.
    X_test : numpy.ndarray
        The test features used for generating predictions.
    y_test : numpy.ndarray
        The actual target values for the test set.
    dates : pandas.Series
        The dates corresponding to the test set.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the following columns:
        - 'Date': The dates of the test set. (index)
        - 'Actual': The actual target values.
        - 'Predicted': The model's predictions.
        - 'Actual Change': The change in actual values from the previous day.
        - 'Binary Actual Change': Binary representation of actual value change
          (1 for increase, -1 for decrease, 0 for no change).
        - 'Predicted Change': The change in predicted values from the previous day.
        - 'Binary Predicted Change': Binary representation of predicted value change
          (1 for increase, -1 for decrease, 0 for no change).

    Description
    -----------
    This function predicts the target values for the test set using the provided model.
    It then calculates the daily change in actual and predicted values and converts these
    changes into binary signals. The resulting DataFrame contains the dates, actual values,
    predictions, and both the continuous and binary changes in values.

    The DataFrame is indexed by the 'Date' column and rows with missing values are dropped.
    """

    predictions = model.predict(X_test)

    df = pd.DataFrame()

    df['Date'] = dates
    df['Actual'] = y_test
    df['Predicted'] = predictions

    df['Actual Change'] = df['Actual'].diff()
    df['Binary Actual Change'] = df['Actual Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    df['Predicted Change'] = df['Predicted'].diff()
    df['Binary Predicted Change'] = df['Predicted Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    df.set_index('Date', inplace=True)
    
    return df.dropna()