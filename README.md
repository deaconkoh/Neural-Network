# Phishing Email Detector using Multi-Input Neural Network
This project demonstrates the construction of a phishing email detector using a multi-input neural network. It combines various email features such as text content (body, subject, and sender), email domain, and the presence of attachments to classify emails into phishing and non-phishing categories. The dataset used is unlabeled, and KMeans clustering is applied to generate pseudo-labels for training.

## Project Structure
The project is organized into the following steps:

1. **Data Extraction:**

    Extracted email components such as body, subject, sender (from), and presence of attachments from raw email data.
   
3. **Feature Engineering:**

      TF-IDF Vectorization: Applied to the combined text content (body, subject, and from) to represent text features.
   
      Frequency Encoding: Used for encoding the email domain.
   
      Binary Encoding: Applied to the presence of attachments to create binary features.
    
4. **Clustering:**

      Sinced the dataset used is unlabeled, I've decided to use KMeans Clustering to generate pseudo-labels for the dataset based on the extracted features.

5. **Model Construction:**

      Built a multi-input neural network to process three input types: text features, domain frequency features, and attachment features.

6. **Training:**

      Trained the model using pseudo-labels generated from clustering.
   
7. **Evaluation:**

      Evaluated the model on test data split from the training data using accuracy as the metric.

## Files
`emails.csv`: The raw unlabeled email dataset.

Available at: [Kaggle](https://www.kaggle.com/code/abhaytomar/starter-the-enron-email-dataset-8c90cc3c-1/input)


## Evaluation and Insights

### Model Architecture
The neural network is built with three distinct inputs:

- Text Content: Processed using a dense layer with 128 units after TF-IDF vectorization.
- Email Domain: Encoded with frequency encoding and processed through a dense layer with 10 units.
- Attachment Information: Binary feature (presence of attachments) processed through a dense layer with 10 units.

The processed features are concatenated and passed through additional dense layers to make the final classification (phishing vs. non-phishing).

### Results

The model is trained using pseudo-labels derived from KMeans clustering, achieving high training accuracy. However, during evaluation, it became evident that overfitting was a significant issue.

### Overfitting
- Training Accuracy: The model achieves near-perfect training accuracy (99.9%) after several epochs, which indicates that the model has learned the training data extremely well.
- Test Accuracy: Although the test accuracy is also high (e.g., 99.64%), the small gap between training and test accuracy suggests that the model may not generalize well to unseen data.

### Causes of Overfitting:
1. Pseudo-Labels: Since the dataset was unlabeled, KMeans clustering was used to generate pseudo-labels. These labels might not be entirely accurate, leading to the model learning patterns that don't generalize well to actual phishing or non-phishing emails.

2. Complexity of the Model: The neural network has multiple dense layers, which might have too many parameters for the relatively small dataset. This can cause the model to overfit the data.

3. Lack of Regularization: No regularization techniques such as dropout or L2 regularization were used, which could have helped prevent overfitting by reducing reliance on certain neurons or large weights.

### Areas for Improvement
1. Better Labeling: Since the dataset is unlabeled, the clustering technique (KMeans) might not provide accurate labels for training. Using actual labeled data would significantly improve the model's performance and generalization ability.

2. Regularization: Adding techniques like dropout or L2 regularization would help prevent overfitting.

3. Dropout: Randomly "drops" units during training to prevent over-reliance on specific neurons.
    - L2 Regularization: Penalizes large weights, helping to control model complexity.
    - Early Stopping: Implement early stopping to stop training when the model's validation accuracy stops improving, preventing overfitting.

4. Simplify the Model: Reducing the number of dense layers and parameters could help prevent overfitting on small datasets.

5. Cross-Validation: Implementing cross-validation would give a more robust measure of how well the model generalizes.

### Future Plans
- Explore different clustering techniques to generate more accurate pseudo-labels.
- Experiment with different model architectures, such as using convolutional layers for the text features or recurrent layers to capture temporal dependencies.
- Apply regularization techniques and early stopping to prevent overfitting.
- Incorporate a mixture of a labeled dataset for a semi-supervised learning which can create a more reliable evaluation of the phishing detection model.

## Conclusion
This project serves as a foundational exploration of building a phishing email detector with a multi-input neural network. While overfitting and other challenges arose, it has been a valuable learning experience. Future efforts will focus on refining the model, improving data representation, and addressing overfitting to enhance performance. This is just the beginning, with many exciting opportunities for further exploration and growth in the field of neural networks!


