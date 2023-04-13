import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QLabel, QLineEdit, QDesktopWidget


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
        self.status_label.resize(760, 40)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("background-color: white; color: black; border: 1px solid black;")

        # Set up the logistic regression model
        self.model = joblib.load("l_reg.joblib")

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

                # Apply the logistic regression model to the input data
                input_data["Action"] = self.model.predict(input_data[["Time (s)", "Acceleration x (m/s^2)", "Acceleration y (m/s^2)",
                                                                        "Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]])
                # Save the output file
                output_file_path = input_file_path[:-4] + "_output.csv"
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
                # Load the input file
                input_data = pd.read_csv(input_file_path)

                # Create the plot
                fig, ax = plt.subplots()
                ax.plot(input_data["Time (s)"], input_data["Acceleration x (m/s^2)"], label="Acceleration x")
                ax.plot(input_data["Time (s)"],
                input_data["Acceleration y (m/s^2)"], label="Acceleration y")
                ax.plot(input_data["Time (s)"], input_data["Acceleration z (m/s^2)"], label="Acceleration z")
                ax.plot(input_data["Time (s)"], input_data["Absolute acceleration (m/s^2)"], label="Absolute acceleration")
                ax.plot(input_data["Time (s)"], input_data["Action"], label="Action")
                ax.legend()
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Acceleration (m/s^2)")
                ax.set_title("Acceleration vs Time")

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