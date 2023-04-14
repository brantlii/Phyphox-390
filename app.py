import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, variation
import joblib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QLabel, QLineEdit, QDesktopWidget


def down_sample(arr, rate):
    # Given a numpy array keep 1 out of every X rows
    d1 = []

    if not isinstance(rate, int):
        print("Please specify an integer downsampling rate with the rate arg")
    else:
        d1 = arr[0::rate]
    return d1

def vec_seg1(array, sub_window_size, overlap: float = 0, clearing_time_index=None, max_time=None, verbose=False):
    if clearing_time_index is None:
        clearing_time_index = sub_window_size

    if max_time is None:
        max_time = array.shape[0]

    stride_size = int(((1 - overlap) * sub_window_size) // 1)
    # print(stride_size)
    start = clearing_time_index - sub_window_size

    sub_windows = (start + np.expand_dims(np.arange(sub_window_size), 0)
                   # Create a rightmost vector as [0, V, 2V, ...].
                   + np.expand_dims(np.arange(max_time - sub_window_size + 1, step=stride_size), 0).T)

    lost = (array.shape[0] - sub_windows[-1, -1] - 1) / array.shape[0]
    last_valid_index = sub_windows[-1, -1]

    if verbose is True:
        print("Last valid index: ", sub_windows[-1, -1])
        print("Data loss due to segmentation: ", last_valid_index)

    # Adapted from the work of Syafiq Kamarul Azman, Towards Data Science
    return array[sub_windows], lost, last_valid_index


def pre_process(data_in):
    # data_in = down_sample(data_in, 5)
    # Convolution will reduce rows by (window_Size - 1)
    rows = data_in.shape[0] - 10 + 1
    cols = data_in.shape[1]

    print("convolution sizing")

    # Ignore the time column
    data_out = np.zeros((rows, cols - 1))
    print("Time ignored")

    # Apply SMA
    for ii in range(cols - 1):
        data_out[:, ii] = np.convolve(data_in.iloc[:, ii + 1], (np.ones(10) / 10), mode='valid')

    data_in = data_in.iloc[0:rows, :]
    print("SMA and input truncation 1 completed")

    # Segment into 500 point chunks
    data_out, _, last = vec_seg1(np.array(data_out), 500)
    print("Data segmented")

    # Segmentation may incur data loss
    data_in = data_in.iloc[0:(last + 1), :]
    print(data_in.shape)
    print(data_out.shape)
    print("input truncation 2 to match viable features")

    return data_in, data_out


def extract_features(raw_data):
    features = pd.DataFrame()
    print('in features')
    x_data, _, _ = vec_seg1(raw_data[:, :, 0].T, 50)
    y_data, _, _ = vec_seg1(raw_data[:, :, 1].T, 50)
    z_data, _, _ = vec_seg1(raw_data[:, :, 2].T, 50)
    a_data, _, _ = vec_seg1(raw_data[:, :, 3].T, 50)

    # Extract the features
    for ti, dat in zip(['x', 'y', 'z', 'a'], [x_data, y_data, z_data, a_data]):
        features['mean' + ti] = np.mean(np.mean(dat, 1), 0)
        features['var' + ti] = np.mean(np.var(dat, 1), 0)
        features['median' + ti] = np.mean(np.median(dat, 1), 0)
        features['std' + ti] = np.mean(np.std(dat, 1), 0)
        features['kurt' + ti] = np.mean(kurtosis(dat, axis=1, fisher=False), 0)
        features['maxim' + ti] = np.mean(np.nanmax(dat, 1), 0)
        features['minim' + ti] = np.mean(np.nanmin(dat, 1), 0)
        features['ptp' + ti] = np.mean(np.ptp(dat, 1), 0)
        features['cvar' + ti] = np.mean(variation(dat, axis=1, nan_policy='omit'), 0)

    corrxy = []
    corrxz = []
    corryz = []
    for a, b, c in zip(raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2]):
        corrxy.append(np.correlate(a,b))
        corrxz.append(np.correlate(a, c))
        corryz.append(np.correlate(b, c))

    features['corrxy'] = corrxy
    features['corrxz'] = corrxz
    features['corryz'] = corryz


    print(features.columns)
    print("feature dataframe made")
    # @Brant if the project needs a CSV of the features to be output could you make that happen here / add that to GUI
    # if save is not None:
    #     if not os.path.exists(save):
    #         save.mkdir(parents=True, exist_ok=True)
    #         if verbose:
    #             print("Made Directory: ", save)
    #
    #     save_train = save / 'features_train.csv'
    #     features.to_csv(save_train, index=False)
    #
    #     save_test = save / 'features_test.csv'
    #     features.to_csv(save_test, index=False)

    return features

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("background-color: gray; color: white;")

        # Set up the window
        self.setWindowTitle("The Phyphox-390 Walking/Jumping Classifier")

        self.setGeometry(0, 0, 800, 250)
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

        # Set up the input file path label and button
        self.file_label = QLabel(self)
        self.file_label.setGeometry(175, 3, 600, 50)
        self.file_label.setText("Phyphox-390: An Application for Accelerometer Data Analysis")
        self.file_label.setStyleSheet("font-size: 16px")
        self.input_file_path_label = QLabel("Input file path:", self)
        self.input_file_path_label.move(40, 50)
        self.input_file_path_text = QLineEdit(self)
        self.input_file_path_text.move(150, 55)
        self.input_file_path_text.resize(500, 20)
        self.browse_button = QPushButton("Browse", self)
        self.browse_button.move(670, 50)
        self.browse_button.clicked.connect(self.browse_input_file)

        # Set up the process button
        self.process_button = QPushButton("Process", self)
        self.process_button.move(350, 100)
        self.process_button.clicked.connect(self.process_input_file)

        # Set up the plot button
        self.plot_button = QPushButton("Plot", self)
        self.plot_button.move(350, 140)
        self.plot_button.clicked.connect(self.plot_data)

        # Set up the status label
        self.status_label = QLabel("", self)
        self.status_label.move(20, 190)
        self.status_label.resize(760, 90)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("background-color: white; color: black; border: 1px solid black;")

        # Set up the logistic regression model
        self.model = joblib.load("10_feature.joblib")

    def browse_input_file(self):
        # Open a file dialog to select the input file
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("CSV Files (*.csv)")
        file_dialog.setDefaultSuffix("csv")
        file_path = file_dialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")[0]

        if file_path:
            # Set the input file path in the text box
            self.input_file_path_text.setText(file_path)


    def process_input_file(self):
        # Get the input file path from the text box
        input_file_path = self.input_file_path_text.text()

        if input_file_path.endswith(".csv"):
            try:
                # Load the input file
                input_data = pd.read_csv(input_file_path)
                print("Input file read successfully.")

                input_data,  sanitized_data = pre_process(input_data)
                print("Pre-processing completed successfully.")

                feature_data = extract_features(sanitized_data)
                print("Features extracted successfully.")

                # Apply the logistic regression model to the input data
                input_data['Action'] = np.repeat(self.model.predict(feature_data), 500)


                # Construct the output file path in the same directory as the input file
                output_file_path = os.path.join(os.path.dirname(input_file_path),
                                                os.path.basename(input_file_path)[:-4] + "_output.csv")

                # Save the output file
                input_data.to_csv(output_file_path, index=False)
                self.status_label.setText("Output file saved successfully.")

            except Exception as e:
                self.status_label.setText(f"Error: {e}")
        else:
            self.status_label.setText("Please select a valid CSV file.")

    def plot_data(self):
        # Get the input file path from the text box
        input_file_path = self.input_file_path_text.text()

        if input_file_path.endswith(".csv"):
            try:
                # Load the output file
                output_file_path = os.path.join(os.path.dirname(input_file_path),
                                                os.path.basename(input_file_path)[:-4] + "_output.csv")
                input_data = pd.read_csv(output_file_path)

                # Create the plot
                fig, ax = plt.subplots(2,1)

                ax[0].plot(input_data["Time (s)"], input_data["Acceleration x (m/s^2)"], linewidth=0.8, label="Acceleration x")
                ax[0].plot(input_data["Time (s)"],
                        input_data["Acceleration y (m/s^2)"], linewidth=0.8, label="Acceleration y")
                ax[0].plot(input_data["Time (s)"], input_data["Acceleration z (m/s^2)"], linewidth=0.8, label="Acceleration z")
                ax[0].plot(input_data["Time (s)"], input_data["Absolute acceleration (m/s^2)"], linewidth=0.8,
                        label="Absolute acceleration")

                ax[0].legend()
                ax[0].set_xlabel("Time (s)")
                ax[0].set_ylabel("Acceleration (m/s^2)")
                ax[0].set_title("Acceleration vs Time")

                ax[1].plot(input_data["Time (s)"], input_data["Action"], linewidth=0.8, label="Action")
                ax[1].legend()
                ax[1].set_xlabel("Time (s)")
                ax[1].set_ylabel("Action")
                ax[1].set_title("Action Prediction vs Time")

                fig.set_layout_engine(layout='tight')
                plt.show()
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
        else:
            self.status_label.setText("Please select a valid CSV file.")

if __name__ == "__main__":
    # Create the application and set font
    app = QApplication(sys.argv)
    font = app.font()
    font.setFamily("Helvetica")
    app.setFont(font)

    # Create and show the main window
    window = MainWindow()
    window.show()

    # Start the event loop
    sys.exit(app.exec_())

