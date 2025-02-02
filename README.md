# SENTIMENT-ANALYSIS-WITH-NLP

**COMPANY** : CODTECH IT SOLUTIONS

**NAME** : GAURAV RAMASUBRAMANIAM

**INTERN ID** : CT08JUR

**DOMAIN** : MACHINE LEARNING

**DURATION** : 4 WEEKS

**MENTOR** : NEELA SANTOSH

# DESCRIPTION

# Reddit Vaccine Myths Sentiment Analysis

## Project Overview
This project aims to analyze sentiment in Reddit posts related to vaccine myths using Natural Language Processing (NLP) techniques. By leveraging machine learning, we classify posts as either **positive** (supportive of vaccines) or **negative** (against vaccines) based on their sentiment. We use the **Reddit Vaccine Myths dataset** from Kaggle for training and evaluation.

## Dataset
**Dataset Name:** [Reddit Vaccine Myths](https://www.kaggle.com/datasets/gpreda/reddit-vaccine-myths)  
**Description:** This dataset contains posts discussing vaccine myths on Reddit. It includes fields such as:
- `title`: The title of the Reddit post
- `body`: The content of the Reddit post
- `score`: The post’s Reddit score (upvotes - downvotes)
- `upvote_ratio`: Ratio of upvotes to total votes
- `subreddit`: The subreddit where the post was made

## Project Workflow
### 1. Data Preprocessing
- Combine `title` and `body` into a single text column.
- Convert text to lowercase and remove special characters.
- Remove stopwords using NLTK's stopword list.

### 2. Feature Extraction
- Convert cleaned text into numerical features using **TF-IDF Vectorization** (Top 5000 features selected).

### 3. Sentiment Labeling
- Define sentiment based on `score`: 
  - **Positive Sentiment (1):** `score` > 0
  - **Negative Sentiment (0):** `score` ≤ 0

### 4. Model Training & Evaluation
- Split data into **training (80%)** and **testing (20%)** sets.
- Train a **Logistic Regression** model.
- Evaluate model performance using:
  - **Accuracy Score**
  - **Classification Report** (Precision, Recall, F1-score)
  - **Confusion Matrix**
  
### 5. Results & Visualization
- Visualize sentiment distribution using Seaborn.
- Identify top **positive** and **negative** words contributing to predictions based on model coefficients.

## Installation & Usage
### Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```
### Running the Script
1. Download the dataset from Kaggle and place it in your working directory.
2. Run the script:
```bash
python sentiment_analysis.py
```
3. View accuracy, classification report, and visualizations.

## Key Findings
- The sentiment classification model provides insights into the types of words commonly associated with vaccine misinformation or support.
- The dataset shows a diverse range of opinions, highlighting the polarized nature of vaccine discussions online.

## Future Improvements
- Experiment with other ML models (SVM, Random Forest, Deep Learning).
- Use sentiment lexicons to enhance preprocessing.
- Incorporate subreddit-based sentiment trends.

## Author
This project was developed as an NLP-based sentiment analysis task using machine learning. Contributions and improvements are welcome!

# OUTPUT

![Image](https://github.com/user-attachments/assets/81738103-5487-4bf5-98d6-200f5057c2b1)

![Image](https://github.com/user-attachments/assets/e9440ca2-87ad-4be4-ad8e-c1760e9838b4)
