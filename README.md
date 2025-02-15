# Twitter Rumor Detection using Machine Learning and NLP

## Project Overview
This project aims to classify Twitter posts as either **Rumor** or **Non-Rumor** using machine learning and natural language processing (NLP). The dataset consists of tweets labeled as "True," "False," "Non-Rumor," and "Unverified," which are later merged into two categories: **Rumor** (False + Unverified) and **Non-Rumor** (True + Non-Rumor).

## Features & Methodology
- **Exploratory Data Analysis (EDA):**
  - Label distribution visualization
  - Text length distribution
  - Word cloud for common word analysis
  - Correlation between labels and text
- **Text Preprocessing:**
  - Removal of HTML tags, punctuation, emojis, and stopwords
  - Lemmatization for word normalization
  - Converting text data into numerical features using CountVectorizer
- **Model Training & Evaluation:**
  - Splitting data into training and testing sets
  - Training a **KNN,Logistic Regression , Random forest and support vector classifier** model
  - Hyperparameter tuning for performance optimization
  - Evaluating model performance using accuracy, confusion matrix, and classification report
  - Cross-validation to check overfitting
- **Overfitting Handling:**
  - Regularization and hyperparameter tuning to improve generalization
- **Model Saving & Deployment:**
  - Saving the trained model for future use
  - Loading and using the model for predictions

## Technologies Used
- **Programming Language:** Python
- **Libraries & Frameworks:** Pandas, NumPy, Matplotlib, Seaborn, NLTK, Scikit-learn
- **Machine Learning Model:** Support vector classifier 
- **NLP Techniques:** Tokenization, Stopword Removal, Lemmatization, Count Vectorization

## Dataset
The dataset includes two files:
1. `source_tweets.txt`: Contains tweet texts.
2. `label.txt`: Contains corresponding labels.



## Results & Insights
- The model successfully distinguishes rumors from non-rumors with a reasonable accuracy.
- Common words in rumor tweets differ significantly from non-rumors.
- Overfitting was mitigated using cross-validation and hyperparameter tuning.

## Future Improvements
- Experimenting with advanced models like **LSTMs** or **Transformers (BERT)** for improved accuracy.
- Integrating **TF-IDF** instead of CountVectorizer for better feature extraction.
- Deploying the model using Flask for real-time rumor detection.


