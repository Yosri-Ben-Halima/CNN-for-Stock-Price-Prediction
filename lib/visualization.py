import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_predictions(date, y_test, predictions, fig, row=1, col=1):
    """
    Plots the actual and predicted stock prices on a specified subplot of a figure.

    Parameters
    ----------
    date : array-like
        An array or list of dates corresponding to the stock prices.
    y_test : array-like
        An array or list of actual stock prices.
    predictions : array-like
        An array or list of predicted stock prices.
    fig : plotly.graph_objects.Figure
        A Plotly figure object where the traces will be added.
    row : int, optional
        The row index of the subplot where the traces will be plotted. Default is 1.
    col : int, optional
        The column index of the subplot where the traces will be plotted. Default is 1.

    Returns
    -------
    None

    Description
    -----------
    This function adds traces to a Plotly figure to visualize the actual and predicted stock prices.
    It creates two line plots: one for the actual prices and one for the predicted prices.
    The x-axis is labeled as 'Time', and the y-axis is labeled as 'Price'.
    The traces are added to the subplot specified by the `row` and `col` parameters.
    """

    fig.add_trace(
        go.Scatter(x=date, y=y_test, mode='lines', name='Actual Prices'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=date, y=predictions, mode='lines', name='Predicted Prices'),
        row=row, col=col
    )

    fig.update_xaxes(title_text='Time', row=row, col=col)
    fig.update_yaxes(title_text='Price', row=row, col=col)


def plot_mean_percentage_error(date, mpe, fig, row=3, col=1):
    """
    Plots the mean percentage error (MPE) between actual and predicted values on a specified subplot of a figure.

    Parameters
    ----------
    date : array-like
        An array or list of dates corresponding to the MPE values.
    mpe : array-like
        An array or list of mean percentage errors (MPE) values.
    fig : plotly.graph_objects.Figure
        A Plotly figure object where the trace will be added.
    row : int, optional
        The row index of the subplot where the trace will be plotted. Default is 3.
    col : int, optional
        The column index of the subplot where the trace will be plotted. Default is 1.

    Returns
    -------
    None

    Description
    -----------
    This function adds a trace to a Plotly figure to visualize the mean percentage error (MPE) over time.
    It creates a line plot showing the percentage difference between the actual and predicted values.
    The x-axis is labeled as 'Time', and the y-axis is labeled as 'MPE (%)'.
    The trace is added to the subplot specified by the `row` and `col` parameters.
    """

    fig.add_trace(
        go.Scatter(x=date, y=mpe, mode='lines', name='MPE (%)'),
        row=row, col=col
    )

    fig.update_xaxes(title_text='Time', row=row, col=col)
    fig.update_yaxes(title_text='MPE (%)', row=row, col=col)


def plot_training_history(history, fig, row=3, col=2):
    """
    Plots the training and validation loss over epochs on a specified subplot of a figure.

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned from model training, containing loss and validation loss data.
    fig : plotly.graph_objects.Figure
        A Plotly figure object where the traces will be added.
    row : int, optional
        The row index of the subplot where the traces will be plotted. Default is 3.
    col : int, optional
        The column index of the subplot where the traces will be plotted. Default is 2.

    Returns
    -------
    None

    Description
    -----------
    This function adds two traces to a Plotly figure to visualize the training and validation loss over epochs.
    It creates line plots showing how the loss and validation loss change as training progresses.
    The x-axis is labeled as 'Epoch', and the y-axis is labeled as 'Loss'.
    The traces are added to the subplot specified by the `row` and `col` parameters.
    """

    epoch = np.arange(len(history.history['loss']))
    fig.add_trace(
        go.Scatter(x=epoch, y=history.history['loss'], mode='lines', name='Training Loss'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=epoch, y=history.history['val_loss'], mode='lines', name='Validation Loss'),# line=dict(dash='dash')),
        row=row, col=col
    )
    fig.update_xaxes(title_text='Epoch', row=row, col=col)
    fig.update_yaxes(title_text='Loss', row=row, col=col)


def plot_price_moves(df, fig, row=2, col=1):
    """
    Plots a colormap of actual vs. predicted price movements, using different colors for up and down movements.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns for 'Binary Actual Change' and 'Binary Predicted Change', 
        which represent the binary price movements.
    fig : plotly.graph_objects.Figure
        A Plotly figure object where the traces will be added.
    row : int, optional
        The row index of the subplot where the traces will be plotted. Default is 2.
    col : int, optional
        The column index of the subplot where the traces will be plotted. Default is 1.

    Returns
    -------
    None

    Description
    -----------
    This function adds two scatter plots to a Plotly figure to visualize the actual and predicted price movements. 
    The price movements are color-coded as green for 'Up', red for 'Down', and grey for 'No Change'. 
    Actual movements are plotted on the top row of the subplot, and predicted movements are plotted on the bottom row. 
    The x-axis represents time, and the y-axis represents binary price movements. The colors and symbols of the markers 
    are used to differentiate between the actual and predicted movements.
    """

    act_colors = df['Binary Actual Change'].map({1: 'green', -1: 'red', 0: 'grey'})
    act_symbols = df['Binary Actual Change'].map({1: 'square', -1: 'square', 0: 'square'})
    pred_colors = df['Binary Predicted Change'].map({1: 'green', -1: 'red', 0: 'grey'})
    pred_symbols = df['Binary Predicted Change'].map({1: 'square', -1: 'square', 0: 'square'})

    fig.add_trace(go.Scatter(
        x=df.index,
        y=[1] * len(df),
        mode='markers',
        marker=dict(
            size=15,
            color=act_colors,
            symbol=act_symbols,
            line=dict(color='black', width=1),
        ),
        name='Actual'),
        row=row, col=col
        )

    fig.add_trace(go.Scatter(
        x=df.index,
        y=[0] * len(df), 
        mode='markers',
        marker=dict(
            size=15,
            color=pred_colors,
            symbol=pred_symbols,
            line=dict(color='black', width=1),
        ),
        name='Predicted'),
        row=row, col=col
        )

    fig.update_xaxes(
        title_text='Time',
        showgrid=False,
        zeroline=False,
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text='Binary Change',
        tickvals=[0, 1],  # Adjusted to include -1 for down moves
        ticktext=['Predicted', 'Actual'],  # Updated labels for clarity
        showgrid=False,
        zeroline=False,
        range=[-0.5, 1.5],  # Adjusted range to include -1 for down moves
        row=2,
        col=1
    )


def plot_monitoring_dashboard(history, predictions_df, ticker):
    """
    Creates and displays a dashboard with various plots to monitor the model's performance and predictions.

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned from the model training process, containing training and validation metrics.
    predictions_df : pandas.DataFrame
        DataFrame containing columns 'Actual', 'Predicted', and 'MPE' (Mean Percentage Error), 
        along with an index representing the dates of predictions.
    ticker : str
        The stock ticker symbol used to fetch and predict stock prices.

    Returns
    -------
    None

    Description
    -----------
    This function generates a Plotly dashboard with multiple subplots to visualize different aspects of model performance:
    - The first subplot shows the comparison of actual and predicted stock prices.
    - The second subplot displays the actual vs. predicted price movements.
    - The third subplot illustrates the Mean Percentage Error (MPE) between the actual and predicted prices.
    - The fourth subplot presents the training and validation loss curves.

    The dashboard is organized into a 3x2 grid layout, with customized subplot titles and axis configurations.
    The layout of the entire figure is set with a title that includes the stock ticker symbol, and various visual 
    enhancements such as unified hover mode and legend positioning.
    """
    
    fig = make_subplots(
        rows=3, cols=2, 
        column_widths=[0.5, 0.5],  
        subplot_titles=("Prediction vs. Actual Stock Price", "Prediction vs. Actual Stock Price Movement", "Mean Percentage Error", "Model Loss"),
        specs=[[{"colspan": 2}, None],[{"colspan": 2}, None], [ {"colspan": 1}, {"colspan": 1}]],  
        vertical_spacing=0.1
    )

    plot_predictions(predictions_df.index, predictions_df['Actual'],predictions_df['Predicted'],fig)
    plot_price_moves(predictions_df, fig)
    plot_mean_percentage_error(predictions_df.index, predictions_df['MPE'] ,fig)
    plot_training_history(history,fig)

    fig.update_layout(title_text=f"Model Training Dashboard | Stock Ticker: {ticker}",
                        hovermode='x unified',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        margin=dict(l=40, r=40, t=40, b=40),
                        height= 1000)

    fig.show()
