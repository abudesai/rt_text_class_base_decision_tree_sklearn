Decision Tree Classifier in SciKitLearn Using Tf-IDF preprocessing for Text Classification - Base problem category as per Ready Tensor specifications.

- decision tree
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- text classification

This is a Text Classifier that uses a Decision Tree Classifier implemented through SciKitLearn.

The classifier starts by creating decision trees that look at whether the variables meet certain thresholds to create leaf nodes that allow the classifier to assign the sample into a specific class.

The data preprocessing step includes tokenizing the input text, applying a tf-idf vectorizer to the tokenized text, and applying Singular Value Decomposition (SVD) to find the optimal factors coming from the original matrix. In regards to processing the labels, a label encoder is used to turn the string representation of a class into a numerical representation.

Hyperparameter Tuning (HPT) is conducted by finding the optimal number of samples required to split an internal node as well as the optimal number of samples required to be at a leaf node.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as clickbait, drug, and movie reviews as well as spam texts and tweets from Twitter.

This Text Classifier is written using Python as its programming language. SciKitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.
