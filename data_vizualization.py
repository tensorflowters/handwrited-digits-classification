import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc


"""
Function:
    Displays an image from the dataset with its corresponding digit label

Args:
    dataset (numpy.ndarray): The dataset containing images.
    labels (numpy.ndarray): The associated labels in the dataset.
"""
def plot_image(dataset, labels):
    index = int(input("\nEnter the index of the image you want to display: "))
    
    if 0 <= index < len(dataset):
        plt.figure(figsize=(12, 10))

        plt.imshow(dataset[index], cmap=plt.cm.binary)

        # Set the grid to the digit image dimension in pixels
        plt.gca().set_xticks([x - 0.5 for x in range(1, 28)], minor=True)
        plt.gca().set_yticks([y - 0.5 for y in range(1, 28)], minor=True)
        plt.grid(which='minor', linestyle='-', linewidth=0.5, color='black')

        plt.xticks([])
        plt.yticks([])

        plt.xlabel(f"Digit label: {labels[index]}")

        plt.show()
    else:
        print(f"\nInvalid index. Please enter a number between 0 and {len(dataset) - 1}.")


"""
Definition:
    Distribution refers to the frequency or proportion of different classes or categories within the dataset.

Function:
    Display 2 charts representing the distribution of each digit category in the training and testing dataset.

Args:
    y_train (numpy.ndarray): Digits labels for the training set.
    y_test (numpy.ndarray): Digits labels for the testing set.
"""
def plot_distribution(y_train, y_test):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train)
    plt.title("Digits distribution (training dataset)")
    plt.xlabel("Digits")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.countplot(x=y_test)
    plt.title("Digits distribution (testing dataset)")
    plt.xlabel("Digits")
    plt.ylabel("Frequency")

    plt.show()


"""
Definition:
    The mean refer to a statistical measure of central tendency of the predicted probabilities or scores assigned to each class by the model.
    However, it is important to note that the mean predicted probabilities may not always be the best measure of model performance,
    as it depends on the problem and the specific use case.

Function:
    Display a chart reprsenting the mean occurrence of each label in the training and testing sets.

Args:
    y_train (numpy.ndarray): Digits labels for the training set.
    y_test (numpy.ndarray): Digits labels for the testing set.
"""
def plot_mean(y_train, y_test):
    unique_labels = np.unique(y_train)

    mean_train = [np.mean(y_train == label) * 100 for label in unique_labels]
    mean_test = [np.mean(y_test == label) * 100 for label in unique_labels]

    bar_width = 0.35
    index = np.arange(len(unique_labels))

    plt.figure(figsize=(20, 10))

    plt.bar(index, mean_train, bar_width, label='Training dataset')
    plt.bar(index + bar_width, mean_test, bar_width, label='Testing dataset')

    plt.xlabel('Digits')
    plt.ylabel('Mean occurence (%)')

    plt.title('Mean occurence of each digit in their respective dataset')
    plt.xticks(index + bar_width / 2, unique_labels)
    plt.legend()
    plt.tight_layout()

    plt.show()

"""
Function:
    Displays an digit image on which our model has made a prediction.
    Similar to plot_image except that the prediction and validation results are display to

Args:
    i (int): Index of the digit image in the dataset.
    predictions_array (numpy.ndarray): Array of predictions for the digit image.
    true_label (int): Validation test label of the image.
    img (numpy.ndarray): Digit image to be displayed.
"""
def plot_prediction(i, predictions_array, true_label, img):
    plt.figure(figsize=(12, 10))

    true_label, img = true_label[i], img[i]

    # Set the grid to the digit image dimension in pixels
    plt.gca().set_xticks([x - 0.5 for x in range(1, 28)], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, 28)], minor=True)
    plt.grid(which='minor', linestyle='-', linewidth=0.5, color='black')

    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.2f}% ({})".format(predicted_label, 100 * np.max(predictions_array), true_label), color=color)
    
    plt.show()

"""
Definition:
    A confusion matrix, also known as an error matrix, is a visualization tool used to evaluate the performance of a classification model.
    It is a table that summarizes the number of correct and incorrect predictions made by the model, broken down by each class.
    The confusion matrix helps identify patterns or trends in misclassifications, making it easier to understand the model's strengths and weaknesses.

Function:
    Display a chart representing the predicted result confusion maxtrix.

Args:
    labels_true (numpy.ndarray): The true labels for the dataset.
    labels_pred (numpy.ndarray): The predicted labels for the dataset.
"""
def plot_confusion_matrix(labels_true, labels_pred):


    matrix = confusion_matrix(labels_true, labels_pred)

    plt.figure(figsize=(16, 10))

    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt="d", xticklabels=range(10), yticklabels=range(10))
    
    plt.title("Confusion matrix")

    plt.xlabel("Predicted digit")
    plt.ylabel("Actual digit")

    plt.show()

"""
Definition:
    The Receiver Operating Characteristic (ROC) curve is a graphical representation that plots the true positive rate (sensitivity) against the false positive rate (1-specificity) at various threshold levels.
    The true positive rate (TPR) is the proportion of actual positive instances that are correctly identified by the classifier, while the false positive rate (FPR) is the proportion of actual negative instances that are incorrectly identified as positive.
    It helps visualize the trade-off between sensitivity (TPR) and specificity (1-FPR) for a classifier.
    A perfect classifier would have an ROC curve that hugs the top-left corner of the plot, indicating a high true positive rate and a low false positive rate.
    A classifier with no predictive power would lie on the diagonal line, representing a random guess.

    The AUC stands for "Area Under the ROC Curve." It is a single scalar value that measures the classifier's overall performance across all threshold levels.
    The AUC ranges from 0 to 1, with a higher value indicating better classifier performance.
    An AUC of 0.5 represents a classifier with no discriminative power, equivalent to random guessing, while an AUC of 1 represents a perfect classifier that makes no mistakes.
    AUC is useful for comparing different classifiers, as it takes into account both the true positive rate and false positive rate.
    It is also less sensitive to class imbalance, which makes it a popular choice for evaluating classification models in real-world applications where class distribution might be skewed.

Function:
    Display a chart representing the predicted result ROC curve and AUC values for each class category.

Args:
    y_true (numpy.ndarray): The true labels for the dataset.
    y_pred_probs (numpy.ndarray): The predicted probabilities for each class.
"""
def plot_roc_curve(y_true, y_pred_probs):
    y_true_bin = label_binarize(y_true, classes=range(10))
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot the ROC curve
    plt.figure(figsize=(20, 10))

    # Compute the AUC value for each digit category
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Digit {i} (AUC = {roc_auc[i]:.4f})")

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.title('Receiver Operating Characteristic curve')

    plt.legend(loc='lower right')

    plt.show()
