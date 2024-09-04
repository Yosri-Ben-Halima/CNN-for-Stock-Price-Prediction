# CNN for Stock Price Prediction

This project involves using Convolutional Neural Networks (CNNs) to forecast stock prices and predict price movements. The main objectives are to build a CNN model for stock price prediction, evaluate its performance, and visualize results through various metrics and plots.

## 1. Project Structure

- **`analysis_notebook.ipynb`**: Jupyter Notebook that walks through the entire workflow, including data fetching, model training, evaluation, and visualization. 

- **`lib/`**: A directory containing utility modules for metrics, visualization, data preparation, and model building.
  - **`metrics.py`**: Contains functions to compute evaluation metrics like mean percentage error, confusion matrix, F1 score, and accuracy score.
  - **`visualization.py`**: Provides functions for plotting predictions, training history, mean percentage error, and price movements.
  - **`data_preparation.py`**: Includes functions for fetching and preparing stock data for model input.
  - **`model_building.py`**: Contains functions for building and compiling the CNN model, as well as setting up callbacks for training.

## 2. Getting Started

### 2.1. Prerequisites

Ensure you have the following Python packages installed:
- `numpy`
- `pandas`
- `yfinance`
- `tensorflow`
- `plotly`
- `scikit-learn`

You can install the required packages using pip:

```bash
pip install numpy pandas yfinance tensorflow plotly scikit-learn
```

### 2.2. Data Fetching

The `data_preparation.py` module provides the `load_data()` function to download historical stock data for a given ticker symbol and date range. The `prepare_data()` function prepares this data for model training by creating windows of stock prices and normalizing the input features.

### 2.3. Model Building

Use the `build_model()` and `get_callbacks()` functions from the `model_building.py` module to create a CNN model tailored for stock price prediction and define its callbacks for efficient learning. The model includes convolutional layers, pooling layers, and dense layers, as for the callbacks we use the Early Stopping and Learning Rate Reduction. 

### 2.4. Training and Evaluation

The `analysis_notebook.ipynb` notebook demonstrates how to:
1. Fetch and prepare stock data using the `data_preparation.py` functions.
2. Build and train the CNN model using the `model_building.py` functions.
3. Evaluate the model using the `metrics.py` functions:
   - `mean_percentage_error()` to compute the MPE between actual and predicted prices.
   - `binary_evaluation_metrics()` to compute the confusion matrix, F1 score, and accuracy score.
5. Visualize results with the `visualization.py` functions.

### 2.5. Plotting

The `visualization.py` module provides several functions to visualize the model's performance:
- `plot_predictions()`: Shows actual vs. predicted stock prices.
- `plot_price_moves()`: Displays actual and predicted price movements.
- `plot_mean_percentage_error()`: Illustrates the mean percentage error between actual and predicted prices.
- `plot_training_history()`: Plots training and validation loss over epochs.
- `plot_monitoring_dashboard()`: Builds the performance monitoring dashboard of the model.

### 2.6. Example

To see the project in action, open the `analysis_notebook.ipynb` notebook and follow the steps outlined to train and evaluate the model.

## 3. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

