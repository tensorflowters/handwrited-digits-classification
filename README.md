# Tensorflow 2 - Image classification

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://forthebadge.com)

## About the project

These project aim to be an achievable introduction to machine learning image classification algorithm using Tensorflow 2 for beginners. It's based on the Tensorflow documentation and bring more details concerning the installation on your local machine.

_Please refer to the documentation for more details [Tensorflow](https://www.tensorflow.org/overview)_

&nbsp;

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This project assumes that you're using the following configurations:

* Windows 10 or 11
* [WSL 2](https://learn.microsoft.com/fr-fr/windows/wsl/install)
* NVIDIA Graphic card

If you're not metting this requirements, please refer to the appropriate documentation according to your system configurations. Alternatively, if you have a AMD or Intel GPU you can use [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin).

PS: If you're a Mackbook user, maybe it's time to change your computer 😉.

### Installation

1. Check if your NVIDIA drivers are up to date. For that, you can use [Geforce Experience application](https://www.nvidia.com/fr-fr/geforce/geforce-experience/) or directly download them on the [NVIDIA website](https://www.nvidia.com/Download/index.aspx).
2. Check if WSL2 is up to date. If not, you can update it with the following command in a windows promt:

   ```sh
   wsl.exe --update
   ```

3. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) on your Windows distribution. Please follow the instructions untill you've completed the installation.

4. Now, we need to install CUDA Toolkit on WSL2. First, open a WSL2 prompt and remove the old GPG key by enter the following command:

   ```bash
   sudo apt-key del 7fa2af80
   ```

   Then, you can process to the installation by entering the following commands:

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   ```

   ```bash
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   ```

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
   ```

   ```bash
   sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
   ```

   ```bash
   sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
   ```

   ```bash
   sudo apt-get update
   ```

   ```bash
   sudo apt-get -y install cuda
   ```

   _Please refer to the official [documentation](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) for more details concerning this step_

5. Install Miniconda to handle virtual environments in your WSL2 distribution by running the following commands:

   ```bash
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
   ```

   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

   Please follow the installation steps in the Miniconda prompt. After that you can restart your terminal and check the installed conda version by enter the following command:

   ```bash
   conda -V
   ```

   You should see something like:

   ```bash
   conda 23.3.0
   ```

6. Create a virtual env named tensorflow with python 3.9 by running the following command:

   ```bash
   conda create --name tensorflow python=3.9
   ```

7. If not activate, activate the tensorflow env by running the following command:

   ```bash
   conda activate tensorflow
   ```

8. Check the NVIDIA drivers installation:

   ```bash
   nvidia-smi
   ```

   You should see something like:

   ```bash
   Sat Apr  1 04:09:31 2023       
   +---------------------------------------------------------------------------------------+
   | NVIDIA-SMI 530.41.03              Driver Version: 531.41       CUDA Version: 12.1     |
   |-----------------------------------------+----------------------+----------------------+
   | GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                                         |                      |               MIG M. |
   |=========================================+======================+======================|
   |   0  NVIDIA GeForce RTX 3090         On | 00000000:2D:00.0  On |                  N/A |
   |  0%   28C    P8               33W / 420W|   2587MiB / 24576MiB |      4%      Default |
   |                                         |                      |                  N/A |
   +-----------------------------------------+----------------------+----------------------+
                                                                                            
   +---------------------------------------------------------------------------------------+
   | Processes:                                                                            |
   |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
   |        ID   ID                                                             Usage      |
   |=======================================================================================|
   |  No running processes found                                                           |
   +---------------------------------------------------------------------------------------+
   ```

9. Now you need to install CUDA toolkit and some dependencies in the conda environment:

   ```bash
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```

10. Create the config repository access for conda:

    ```bash
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    ```

11. Export the repository path access for conda:

    ```bash
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```

    Restart your terminal and reactivate your conda env before passing to the next step.

12. Now, still in your conda tensorflow env, install tensorflow by running the following commands:

    ```bash
    python3.9 -m pip install --upgrade pip
    ```

    ```bash
    python3.9 -m pip install tensorflow==2.8.0
    ```

    You can test if the installation was success by running:

    ```bash
    python3.9 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

&nbsp;

## Usage

### Trainning

   ```bash
    python3.9 model_training.py
   ```

### Inference

   ```bash
    python3.9 model_inference.py
   ```

### Tips

* [mnist_model.h5](mnist_model.h5) is the train model to format HDF5 format. You can export/import it where ever you want !
* See [exomodel.py](exomodel.py) for more details about loading pre-train model
* [hypertunning_logs](hypertunning_logs) directory is generated ONLY at the first train.
* If you want to reset the hyperparameters search, you just need to remove [hypertunning_logs](hypertunning_logs) directory and restart training
* Feel free to re-adjust the hypermodel as you whish

&nbsp;

## Questions that need to be ask

&nbsp;

### Why use a separate dataset to measure the performance of an algorithm ?

```text
Using a separate dataset is essential to measure the performances of an algorithm 
because it's allowing our model to avoid making predictions on memorized data. 
The aim of a machine learning algorithm is to make accurate predictions on unseen 
data and not only on know data.
```

&nbsp;

### What are your results when you test your algorithm on the same dataset used in training ?

```text
Your results may be too good to be true. Since, your model has already seen this example, it might have memorized theme. 

This issue is known as overfitting. This when the algorithm perform well on training data but poorly on unseen ones. This can lead you to have a model with very limited practical use.

To overcome this issue, it's common practice to split the available data into three separate sets:

- Training set: This is the largest portion of the dataset, used to train the algorithm and adjust its parameters.

- Validation set: This set is used to tune hyperparameters, compare different models, and make decisions about the model architecture during the training      process.

- Test set: This set is used to evaluate the performance of the final algorithm. It should only be used once after the model has been fully trained and tuned, to provide an unbiased estimate of the algorithm's performance on new data.

By separating the data into different sets, you can ensure a more accurate and reliable estimation of your algorithm's performance and its ability to generalize to new, unseen data.
```

&nbsp;

### What are bias and variance ?

```text
Bias and variance are two fundamental sources of error in machine learning models that help us understand the trade-off between model complexity and generalization.

Bias refers to the error introduced by approximating a real-world problem with a simplified model. It measures how far off the model's predictions are from the true values on average. A high bias means the model makes strong assumptions about the data and may not capture the underlying patterns, leading to underfitting. Underfitting occurs when the model is too simple and cannot accurately represent the complexity of the data.

Variance, on the other hand, refers to the error introduced by the model's sensitivity to small fluctuations in the training data. A high variance means the model is overly complex and tries to fit the noise in the training data, leading to overfitting. Overfitting occurs when the model learns not only the underlying patterns in the data but also the random noise, making it perform poorly on new, unseen data.

In general, there is a trade-off between bias and variance in machine learning models. If you increase the model's complexity to reduce bias, you are likely to increase its variance, making it more sensitive to noise in the data. Conversely, if you decrease the model's complexity to reduce variance, you are likely to increase its bias, making it less capable of capturing the true patterns in the data.

The goal in machine learning is to find the right balance between bias and variance, which minimizes the total error and results in a model that generalizes well to new, unseen data. This is often achieved through techniques like regularization, cross-validation, and model selection.
```

&nbsp;

### What is cross validation ?

```text
Cross-validation is a technique used in machine learning to evaluate the performance and generalization ability of a model by partitioning the available data into multiple folds and using these folds for both training and validation. The primary goal of cross-validation is to obtain a more accurate estimate of the model's performance on unseen data and help prevent overfitting.

The most common form of cross-validation is k-fold cross-validation. In k-fold cross-validation, the data is divided into k equally-sized folds. The model is trained and validated k times, each time using a different fold as the validation set and the remaining k-1 folds as the training set. The overall performance is then calculated by averaging the performance metrics across all k iterations.
```

&nbsp;

### What are the main advantages ?

```text
1. More reliable performance estimation: Cross-validation provides a more accurate and robust estimate of the model's performance on unseen data compared to a single train-test split, as it considers multiple training and validation sets.

2. Reduced overfitting: By training and evaluating the model on different subsets of data, cross-validation helps to mitigate overfitting and ensure the model generalizes well to new data.

3. Model selection and hyperparameter tuning: Cross-validation can be used to compare the performance of different models or hyperparameter configurations, helping you select the best model or configuration for your problem.
```

&nbsp;

### When can I use it ?

```text
1. Limited data: Cross-validation is particularly useful when you have limited data, as it maximizes the use of available data for both training and validation.

2. Model selection and hyperparameter tuning: Cross-validation can help you select the best model and hyperparameter configuration by comparing their performance across multiple folds.

3. Ensuring generalization: Cross-validation helps ensure that your model generalizes well to new data by providing a more robust estimation of its performance.

However, keep in mind that cross-validation can be computationally expensive, especially for large datasets or complex models, as it requires training and evaluating the model multiple times. In such cases, other techniques like hold-out validation or stratified sampling might be more appropriate.
```

&nbsp;

### Can you explain why it’s important to normalize (scale) the data when using algos like KNN ?

```text
Normalizing or scaling data is important when using algorithms like k-Nearest Neighbors (KNN) because these algorithms rely on distance metrics to determine the similarity between data points. When features have different scales or units, the distance metric can become dominated by the feature with the largest scale, causing the algorithm to perform poorly or give undue importance to certain features.

By normalizing or scaling the data, you ensure that all features contribute equally to the distance metric, improving the performance and interpretability of the algorithm. Common normalization techniques include min-max scaling, which scales features to a specific range (e.g., [0, 1]), and standardization, which centers the features around the mean with a standard deviation of 1.
```

&nbsp;

### Is it necessary in our specific study case?

```text
Normalization is still a good practice, but for slightly different reasons. The dataset consists of grayscale images of handwritten digits, where each pixel value ranges from 0 to 255. While all features (pixels) are on the same scale, normalizing the data can still provide benefits.

Normalization can help improve the performance and convergence speed of various machine learning algorithms, especially gradient-based optimization algorithms (such as those used in neural networks). Normalizing the pixel values to a smaller range, like [0, 1] or [-1, 1], can make the optimization process more stable, as gradients will be more consistent across different features.
```

&nbsp;

### When you reshaped your image, do you think the order of the columns (that means the order of the pixels) had an importance for the performance of your algorithm?

```text
When using an algorithm like KNN that relies on distance metrics, the order of the pixels in the reshaped image does not have a significant impact on the performance, as long as the order is consistent for all images in the dataset. This is because KNN treats each feature (pixel) independently and calculates distances based on the difference in pixel values between images.

However, the spatial information and relative positions of the pixels do matter for KNN, and reshaping the image into a one-dimensional vector may lead to some loss of this information. Consequently, KNN might not perform as well as algorithms designed to take advantage of spatial relationships, such as convolutional neural networks (CNNs). In the case of CNNs, the order and structure of the pixels are critical for the algorithm's performance, as the convolutional layers are designed to learn spatial patterns and features from the input images.

To summarize, for distance-based algorithms like KNN, the order of the columns (pixels) in the reshaped image does not directly impact the performance as long as it is consistent across the dataset. However, reshaping the image into a one-dimensional vector may lead to a loss of spatial information, which can limit the performance of KNN compared to other algorithms specifically designed to leverage the spatial structure of image data.
```

&nbsp;

### Which metrics measure performance? What does accuracy tell you?

```text
There are several metrics used to measure the performance of machine learning models, depending on the type of problem (classification, regression, etc.) and the specific objectives of the analysis. Some common performance metrics for classification problems include:

1. Accuracy: The proportion of correctly classified instances out of the total instances in the dataset. Accuracy is a simple and intuitive metric, but it can be misleading in cases of imbalanced datasets, where the majority class dominates the minority class. In such cases, a high accuracy can be achieved by simply classifying all instances as the majority class.

2. Precision: The proportion of true positive predictions out of all positive predictions made by the model. Precision is a measure of how well the model correctly identifies positive instances, taking into account false positive predictions.

3. Recall (Sensitivity): The proportion of true positive predictions out of all actual positive instances in the dataset. Recall is a measure of the model's ability to identify all positive instances, taking into account false negatives.

4. F1-Score: The harmonic mean of precision and recall, providing a balance between the two metrics. F1-score is useful when both precision and recall are important to the problem, and it is more informative than accuracy for imbalanced datasets.

5. Area Under the Receiver Operating Characteristic (ROC-AUC) Curve: A measure of the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) at various classification thresholds. A higher ROC-AUC score indicates better classification performance, and it is more robust to class imbalance compared to accuracy.
```

&nbsp;

### Does accuracy penalize more one mistake over another?

```text
Accuracy does not penalize one type of mistake (false positive or false negative) more than another. It simply counts the number of correct predictions and calculates the proportion of those correct predictions out of the total instances. If the cost of different types of mistakes is not equal in your problem, accuracy might not be the most appropriate metric to use. In such cases, you should consider using other performance metrics, like precision, recall, F1-score, or custom loss functions, that better capture the specific objectives and constraints of your problem.
```

&nbsp;

### Can you give an explicit example where accuracy would not be a relevant metric ? In that extreme case, can you propose a more suitable metric ?

```text
Let's consider a real-world example where accuracy might not be the most relevant metric: predicting fraudulent transactions in a credit card transaction dataset.

In this scenario, the dataset is likely to be highly imbalanced, as the vast majority of transactions are legitimate (non-fraudulent), while only a small fraction is fraudulent. If you use accuracy as the performance metric, a model that predicts all transactions as non-fraudulent could still achieve very high accuracy, since it would be correct for all the legitimate transactions. However, this model would be useless in practice, as it would fail to detect any fraudulent transactions.

In this case, precision, recall, and F1-score are more suitable metrics, as they take into account the different types of errors (false positives and false negatives):

- Precision is important because it measures how well the model identifies fraudulent transactions without raising too many false alarms (false positives). A high precision means fewer false alarms, which is important for not inconveniencing customers with false fraud alerts.

- Recall is also important, as it measures the model's ability to identify all actual fraudulent transactions (true positives) without missing any (false negatives). A high recall means that the model is effective at detecting fraud.

- F1-score provides a balance between precision and recall and can be used when both metrics are important to the problem. In the fraud detection scenario, you generally want to minimize both false positives and false negatives, so F1-score might be a more appropriate metric.

Another useful metric for this scenario is the Area Under the Precision-Recall Curve (PR-AUC), which measures the trade-off between precision and recall at different classification thresholds. A higher PR-AUC score indicates better classification performance, especially in imbalanced datasets like the one in this example.
```

&nbsp;

### Do confusion matrix display all informations of algorithms’ learning?

```text
A confusion matrix is a useful tool for visualizing the performance of a classification algorithm, as it provides a summary of the true positive, true negative, false positive, and false negative predictions made by the model. However, it does not display all the information about the algorithm's learning process.

The confusion matrix focuses on the classification results and provides insights into the types of errors made by the model. It allows you to calculate various performance metrics like accuracy, precision, recall, and F1-score. While these metrics help evaluate the performance of the model, they do not provide complete information about the algorithm's learning process.

Information not captured by the confusion matrix includes:

1. Model architecture and parameters: The confusion matrix does not provide any details about the model's structure, complexity, or the specific parameters used during training.

2. Training process and convergence: The confusion matrix does not give information about the training process, such as the number of iterations, learning rate, or how well the model has converged during training.

3. Feature importance: The confusion matrix does not show which features are the most important for making accurate predictions or how they contribute to the model's decisions.

4. Probabilistic outputs: For probabilistic classifiers, the confusion matrix does not display the predicted probabilities for each class, which can provide more granular information about the model's confidence in its predictions.

5. Overfitting or underfitting: While the confusion matrix can help identify poor classification performance, it does not directly indicate whether the model is overfitting or underfitting the data.
```

&nbsp;

### Do you want to have a walk in the forest and observe tree growth ?

```text
Random Forest is an ensemble learning method used for both classification and regression tasks. It works by constructing multiple decision trees during the training phase and aggregating their predictions to produce a more accurate and robust final output.

The main idea behind the Random Forest algorithm is to leverage the power of multiple weak learners (decision trees) to create a strong learner. Here are the key steps in the Random Forest algorithm:

1. Select a random subset of the training data (with replacement) for each tree. This process is called bootstrapping.

2. For each tree, grow the decision tree using the selected subset of data. During the tree construction, at each node, a random subset of features is selected to determine the best split. This introduces randomness and diversity into the tree growth process.

3. Repeat steps 1 and 2 for a specified number of trees, creating an ensemble of decision trees.

4. To make predictions, pass the input data through each tree in the ensemble. For classification tasks, the final output is the class that receives the majority vote from all trees. For regression tasks, the final output is the average of the predictions made by all trees.

The Random Forest algorithm has several advantages, including:

1. Improved accuracy and robustness: By averaging the predictions of multiple decision trees, the Random Forest algorithm reduces the risk of overfitting and improves generalization performance.

2. Handling missing values: Random Forest can handle missing data by considering only the available features for determining the best split at each node.

3. Feature importance: The algorithm can provide an estimate of feature importance by calculating the average reduction in impurity (e.g., Gini impurity or information gain) contributed by each feature across all trees.

However, Random Forests can be computationally expensive, particularly when dealing with large datasets or a large number of trees. Despite this limitation, they are widely used in machine learning due to their accuracy, versatility, and robustness.
```

&nbsp;

## Acknowledgments

Resources you may find helpful !

* [Python 3.9 documentation](https://docs.python.org/3.9/)
* [Tensorflow 2.12.0 documentation](https://www.tensorflow.org/api_docs/python/tf)
* [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)
* [WSL2 installation](https://learn.microsoft.com/fr-fr/windows/wsl/install)
* [CUDA Toolkit Windows](https://developer.nvidia.com/cuda-downloads)
* [CUDA Toolkit GNU/Linux](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
* [Miniconda GNU/Linux repository](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
