import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(file_name):

    with open(file_name) as fil:
        lines = fil.readlines()

        actual_values = []
        predicted_values = []

        for line in lines:
            line = line.strip()
            actual, predicted = line.split(",")
            actual = actual.split(":")[1].strip()
            predicted = predicted.split(":")[1].strip()

            actual_values.append(float(actual))
            predicted_values.append(float(predicted))

        print(actual_values)
        print(predicted_values)

        x = np.array(actual_values)
        y = np.array(predicted_values)
        m, b = np.polyfit(x,y, 1)
        plt.plot(x, m * x + b, color='green', label='Line of Best Fit')
        plt.plot(x, x, 'red',  label='45 degree line')
        plt.scatter(actual_values, predicted_values)

        plt.xlabel('Chronological Age')
        plt.ylabel('Predicted Age')
        # giving a title to my graph
        plt.title(' (b) Validation Set')

        # show a legend on the plot
        plt.legend()
        # function to show the plot
        plt.show()


if __name__ == "__main__":
    # plot_predictions("C:/Users/Aditya/Desktop/Results/5-Better_MAE-5.7 (Made model modifications)/test_result.txt")
    # plot_predictions("C:/Users/Aditya/Desktop/Results/8-Best-MAE-5.1/train.txt")
    plot_predictions("C:/Users/Aditya/Desktop/Results/8-Best-MAE-5.1/val.txt")

