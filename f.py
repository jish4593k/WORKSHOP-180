import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk, ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_data():
    file_path = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        messagebox.showinfo("Info", "Data loaded successfully.")
        return df
    return None

def split_data(df):
    if df is not None:
        features = df.drop(columns=['reading_time'])
        target = df['reading_time']
        return features, target
    return None, None

def linear_regression_scikit(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    predictions = linear_model.predict(X_test)
    return predictions

def linear_regression_keras(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(8, input_dim=X_train.shape[1], activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    predictions = model.predict(X_test)
    return predictions

def draw_distribution_plot(data, title):
    plt.figure(figsize=(8, 5))
    sns.histplot(data, kde=True)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def run_gui():
    root = Tk()
    root.title("Advanced Regression App")

    load_data_button = ttk.Button(root, text="Load Data", command=lambda: load_and_run())
    load_data_button.pack(pady=10)

    quit_button = ttk.Button(root, text="Quit", command=root.destroy)
    quit_button.pack(pady=10)

    def load_and_run():
        df = load_data()
        if df is not None:
            features, target = split_data(df)


            predictions_scikit = linear_regression_scikit(features, target)
            draw_distribution_plot(predictions_scikit, "Linear Regression (scikit-learn)")

          
            predictions_keras = linear_regression_keras(features, target)
            draw_distribution_plot(predictions_keras.flatten(), "Linear Regression (TensorFlow/Keras)")

    root.mainloop()

if __name__ == "__main__":
    run_gui()


