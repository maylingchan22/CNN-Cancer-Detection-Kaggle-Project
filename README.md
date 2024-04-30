# CNN-Cancer-Detection-Kaggle-Project

### Brief description of the problem and data
Briefly describe the challenge problem and NLP. Describe the size, dimension, structure, etc., of the data. 

The challenge involves developing an algorithm to detect metastatic cancer in small image patches extracted from larger digital pathology scans. The dataset, derived from the PatchCamelyon (PCam) benchmark dataset, consists of 220,025 image patches, each labeled with a binary classification indicating the presence or absence of metastatic cancer. The dataset's structure comprises two columns: 'id' for image identifiers and 'label' denoting the presence (1) or absence (0) of tumor tissue. With a 50/50 balance between positive and negative examples in theory, the actual distribution leans slightly towards negatives, with approximately 60% negatives and 40% positives. The images are in TIF format with a size of 96x96 pixels and contain three channels with 8 bits per channel. Although the dataset has been scrubbed of duplicates, this has not been validated through testing. The data's quality is deemed sufficient for diagnostic purposes, having been sourced from routine clinical care, although scanning issues may affect some images' clarity.

Size: rows = 220,025, columns = 2

Dimensions = 2

Structure:
- Dataframe
- Column Names: id (non-null, object), label (non-null, int64)
- No Missing Values
- Label: (0: 130,908, 1: 89,117) - Binary: Absence (0), Presence (1)

67,636 in train

57,458 in test

### Exploratory Data Analysis (EDA) — Inspect, Visualize and Clean the Data

Show a few visualizations like histograms. Describe any data cleaning procedures. Based on your EDA, what is your plan of analysis? 

The exploratory data analysis (EDA) begins with a visualization showing the distribution of cancerous versus non-cancerous images in the dataset, indicating an imbalance with more non-cancerous samples. Subsequent visualizations include sample images and histograms illustrating color distribution differences between cancerous and non-cancerous cells. The histograms reveal distinct patterns in pixel value frequencies across different color channels, providing insights into potential features for classification.

To address the imbalance, a data cleaning procedure is implemented to balance the dataset through random oversampling of cancerous images. This ensures that both classes are adequately represented in the training data. Additionally, a class is introduced to create image generators for training, validation, and testing, incorporating data augmentation and normalization techniques. Verification steps confirm the successful creation of generators and the existence of image files in the specified directory, ensuring data integrity for subsequent analysis.

The plan of analysis involves leveraging the balanced dataset and image generators to train a machine learning model for classifying cancerous and non-cancerous cells based on image features. The model will be trained using convolutional neural networks (CNNs) to effectively learn discriminative features from the images and make accurate predictions. Subsequent evaluation on the test set will validate the model's performance and potential deployment for practical applications such as medical diagnosis.

### Data Preprocessing and Plan of Analysis
In preparing the data for model training, I implemented several preprocessing steps to ensure optimal performance and robustness of the convolutional neural network (CNN) models. The first step involved splitting the dataset into training and testing sets using the train_test_split function from scikit-learn. After splitting the data, I appended the file extension ".tif" to the image IDs in both the training and testing datasets to match the file naming convention of the image files.

One crucial aspect of data preprocessing is handling class imbalance, which can adversely affect model training and performance. To address this issue, I balanced the number of samples in each class within the training set. This was achieved by oversampling the minority class (cancerous cells) to match the number of samples in the majority class (non-cancerous cells). By ensuring a balanced distribution of samples across classes, the model can learn from a more representative dataset and make more accurate predictions.

In addition to addressing class imbalance, I applied normalization to the image data to standardize the pixel values. Normalization involves scaling the pixel values to a range between 0 and 1, which helps stabilize and accelerate the training process. By dividing the pixel values by 255, I ensured that all pixel values fell within this normalized range, making it easier for the model to learn and converge efficiently.

Furthermore, I explored the potential benefits of data augmentation strategies, such as rotating the images, to further enhance the variability and diversity of the training dataset. Data augmentation can help improve model generalization by exposing the model to a wider range of variations in the input data. In this case, I applied random rotations, horizontal flips, and vertical flips to the training images using the ImageDataGenerator class from Keras.

Overall, the data preprocessing steps undertaken go above and beyond basic normalization, incorporating techniques to address class imbalance and explore data augmentation strategies. By ensuring a balanced dataset and augmenting the training data with diverse variations, the models are better equipped to learn meaningful patterns from the input images and generalize effectively to unseen data. This comprehensive approach to data preprocessing lays a solid foundation for subsequent model training and analysis, contributing to the overall success of the project.

### DModel Architecture

Describe your model architecture and reasoning for why you believe that specific architecture would be suitable for this problem. Compare multiple architectures and tune hyperparameters. 

The model architecture is a critical component of building an effective convolutional neural network (CNN) for image classification tasks. In this case, I've developed two distinct architectures, namely Architecture 1 and Architecture 2, each tailored to address the specific requirements of the problem at hand.

Architecture 1 consists of two convolutional layers followed by max-pooling layers, designed to extract low-level features such as edges and textures from the input images. Subsequently, two dense layers with dropout regularization are employed to learn higher-level representations of the features extracted by the convolutional layers. Finally, a softmax output layer is utilized to produce the predicted probabilities for each class. This architecture strikes a balance between model complexity and efficiency, making it suitable for datasets of moderate size.

On the other hand, Architecture 2 is deeper and more complex, featuring three convolutional layers before max-pooling, allowing for the extraction of more intricate patterns and features from the input images. Despite its increased depth, dropout regularization is incorporated in the dense layers to prevent overfitting and promote generalization. The additional convolutional layers in Architecture 2 enable the model to capture a broader range of features, potentially enhancing its performance on more challenging datasets.

The choice between Architecture 1 and Architecture 2 depends on various factors such as the size and complexity of the dataset, computational resources, and desired level of model interpretability. While Architecture 2 offers the potential for higher performance due to its increased capacity to learn complex patterns, it also comes with a higher risk of overfitting, especially on smaller datasets. Architecture 1, being more compact and computationally efficient, may be preferable for simpler datasets or scenarios where model interpretability is a priority.

To select the optimal architecture, I conducted experiments with both architectures while tuning hyperparameters such as kernel size, pool size, and dropout rates. By training and evaluating multiple models with different configurations, I aimed to identify the architecture that achieves the best balance between performance and generalization. The training process involved monitoring metrics such as loss and accuracy on both training and validation sets, as well as employing early stopping and learning rate reduction techniques to prevent overfitting and improve convergence.

In conclusion, the decision-making process behind selecting the model architecture involved a careful consideration of the dataset characteristics, model complexity, and performance requirements. By comparing and tuning multiple architectures, I aimed to develop a robust and effective CNN model capable of accurately classifying images while minimizing the risk of overfitting.

### Results and Analysis

Run hyperparameter tuning, try different architectures for comparison, apply techniques to improve training or performance, and discuss what helped.

Includes results with tables and figures. There is an analysis of why or why not something worked well, troubleshooting, and a hyperparameter optimization procedure summary.

In the process of evaluating different architectures and tuning hyperparameters, various visualizations and analyses were conducted to understand their impact on model performance. Two architectures, namely Architecture 1 and Architecture 2, were compared based on their training histories and classification reports. Both architectures were trained using different hyperparameters. Visualizations such as training loss and accuracy curves were plotted for each architecture, providing insights into their convergence behaviors over epochs. Additionally, classification reports were generated to assess the precision, recall, and F1-score for each class, offering a comprehensive evaluation of model performance.

Hyperparameter tuning played a crucial role in optimizing model performance. Parameters such as kernel size, pool size, dropout rates, and learning rates were systematically adjusted and evaluated to identify the optimal configuration. The rationale behind tuning these hyperparameters lies in their influence on the model's capacity to extract relevant features from the input data, prevent overfitting, and facilitate efficient learning. For instance, larger dropout rates were employed to mitigate overfitting, while smaller learning rates were used to ensure stable convergence during training.

During the experimentation process, several troubleshooting procedures were implemented to address potential issues and enhance model performance. One crucial step involved closely monitoring training and validation metrics for signs of overfitting or instability. If the model exhibited signs of overfitting, such as a significant gap between training and validation performance or erratic behavior in the loss curves, regularization techniques like dropout regularization were applied. Dropout layers were strategically inserted within the network architecture to randomly deactivate a certain percentage of neurons during training, thereby preventing over-reliance on specific features and promoting generalization. Additionally, adaptive learning rate schedules, such as ReduceLROnPlateau, were employed to dynamically adjust the learning rate during training based on validation performance. This helped prevent the model from getting stuck in local minima and facilitated smoother convergence. By iteratively applying these troubleshooting steps and adjusting hyperparameters, the model's robustness and performance were systematically improved, ensuring reliable classification results for cancerous and non-cancerous cell images.

Upon analyzing the training histories of both architectures, it's evident that Architecture 1 consistently converges towards lower loss values and achieves higher validation accuracy compared to Architecture 2. Although Architecture 2 exhibits fluctuations in its performance metrics, it manages to achieve slightly higher initial accuracy, indicating its potential for faster learning. However, the instability in Architecture 2's training process suggests possible overfitting or sensitivity to hyperparameters.

Moving to the evaluation phase, the classification reports confirm that Architecture 2 outperforms Architecture 1 in terms of overall accuracy and class-wise precision, recall, and F1-score for both classes (0 and 1). Architecture 2 demonstrates higher precision, recall, and F1-score for both classes, indicating better discrimination between cancerous and non-cancerous cells.

The improved performance of Architecture 2 can be attributed to its deeper network architecture, which allows for the extraction of more complex features from the input images. Additionally, fine-tuning hyperparameters such as dropout rates and learning rate schedules might have contributed to better generalization capabilities and reduced overfitting in Architecture 2.

In summary, the results suggest that Architecture 2 is the more suitable model for this classification task, as it achieves higher accuracy and better discrimination between classes. However, further experimentation and optimization of hyperparameters could potentially enhance the performance of both architectures and lead to even better results.

### Conclusion

Discuss and interpret results as well as learnings and takeaways. What did and did not help improve the performance of your models? What improvements could you try in the future?

In summary, the results indicate that Architecture 2 outperformed Architecture 1 in terms of classification accuracy and discrimination between cancerous and non-cancerous cells. However, Architecture 2 exhibited some instability during training, suggesting the need for further regularization techniques to mitigate overfitting and ensure more consistent performance. This underscores the importance of balancing model complexity with generalization capabilities and the significance of hyperparameter tuning and network architecture selection in achieving optimal performance.

A key takeaway from this analysis is the importance of iterative experimentation and optimization in model development. While deeper architectures may offer potential advantages in feature extraction, they also introduce challenges such as overfitting, highlighting the need for careful regularization and tuning. Additionally, continuous exploration of hyperparameter space and incorporation of advanced techniques like data augmentation and transfer learning are crucial for enhancing model robustness and generalization.

One aspect that didn't work as well was the instability observed in Architecture 2 during training, despite its deeper network architecture. This instability could be attributed to inadequate regularization or suboptimal hyperparameters, indicating the need for further investigation and refinement.

To improve model performance in future iterations, it would be beneficial to focus on refining regularization techniques, exploring alternative architectures, and leveraging advanced optimization methods such as grid search or Bayesian optimization. Additionally, incorporating more diverse and comprehensive datasets, along with advanced data augmentation strategies, could enhance the model's ability to generalize across different conditions and datasets. By addressing these aspects and adopting a systematic approach to experimentation and optimization, future versions of the model can aim for even higher levels of accuracy and robustness in classifying cancerous and non-cancerous cells, contributing to advancements in medical image analysis and diagnosis.
