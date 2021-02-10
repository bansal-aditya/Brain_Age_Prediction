import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

data_shape = (121, 145, 121)

def plot_distribution(train, val, test):
    train.Age.hist(bins=25, label='Train')
    test.Age.hist(bins=25, label='Test')
    val.Age.hist(bins=25, label='Validation')
    plt.xlabel('Age')
    plt.ylabel('Nr of images')
    plt.legend()
    plt.show()


def plot_loss(h):
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('ResNet Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')

    plt.show()

    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('ResNet MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')

    plt.show()


def plot_validation_n_tes_prediction(val_prediction, val, test_prediction, test):
    predictions = val_prediction[:, 0]
    yVal = val.Age.values
    print('Validation R^2: ', r2_score(yVal, predictions))
    print('Validation MAE: ', mean_absolute_error(yVal, predictions))
    y_range = np.arange(np.min(yVal), np.max(yVal))
    plt.scatter(yVal, predictions, label='T1 Prediction')
    plt.plot(y_range, y_range, c='black', ls='dashed', label='45 deg line')
    plt.xlabel('Age')
    plt.ylabel('Predicted Age')
    plt.legend()
    plt.show()

    predictions = test_prediction[:, 0]
    yTest = test.Age.values

    print('Test R^2: ', r2_score(yTest, predictions))
    print('Test MAE: ', mean_absolute_error(yTest, predictions))
    y_range = np.arange(np.min(yTest), np.max(yTest))
    plt.scatter(yTest, predictions, label='T1 prediction')
    plt.plot(y_range, y_range, c='black', ls='dashed', label='45 deg line')
    plt.xlabel('Age')
    plt.ylabel('Predicted Age')
    plt.legend()
    plt.show()


def plotData(batch_data, subjectNr=1, c=5, d=12, nSlices=10, figsize=(20, 10), inputShape=data_shape, augment=False,
             training=False, resize=True):
    fig, axs = plt.subplots(1, nSlices, figsize=figsize)
    for i in range(nSlices):
        axs[i].imshow(batch_data[0, :, :, d * (i + c), 0], interpolation="spline16", cmap='gray')
    plt.show()


def bar_graph(labels, scores):

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, scores, width, label='MAE')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MAE')
    ax.set_title('MAE by age group')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)

    fig.tight_layout()

    plt.show()