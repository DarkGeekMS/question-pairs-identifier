# Introduction and Problem
Quora is an online platform to connect with people, ask questions and get answers. With over 100 million people visit Quora every month, there must be a huge amount of duplicate questions that have the same words or carry the same meaning. This can be very confusing for the answer seekers and results in a longer search time.  The huge amount of questions on Quora urges the usage of Big Data Analytics techniques. Our systemoffers a way to classify questions based on their content, inorder to relate duplicate questions with each others for easier and quicker search.

# Dataset
We are going to use the dataset from [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) Kaggle’s competition.The dataset consists of questions pairs with labels for whether they are related or not.

# Dataset Analysis
- Number of question pairs: 404290
- Number of unique questions: 537933
- Number of questions that appear multiple times: 111780
- Histogram shows that most of the questions are not duplicated:

![image](https://drive.google.com/uc?export=view&id=1PPl_K65m4oGMML0sEHGLOVgir_Fr_wHb)
- Data is unbalanced as Duplicate pairs percentage: 36.92%, Non-Duplicate pairs percentage: 63.08% 
- We re-balanced the dataset by re-sample more samples from duplicate pairs.
- Data has words that are repeated frequently in the questions and are not stop words.
- Here's a word cloud graph representing the most frequent words:

![image](https://drive.google.com/uc?export=view&id=1HyLvXBQKscZxTqvfahWfO8RIz5T33RoT)
- We may need TfIdf feature to caption these words and not to be biased to them.
- From the word cloud above, it seems like most of the collected questions are from Indian people, as India is the most common word. This may indicate that our features may be biased to Indian people.
- Words `Best` and `Way` are frequent, which means that many questions are asking about best way to do something.
- Word `Difference` is frequent as well, which means that many questions are asking about differences between things.

# Feature extractor and Classical classifiers

# Available Classifiers
- Naive Bayes
- Logisitic Regression
- XGBoost

# To Train 

1. Download [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/data?select=train.csv.zip).
2. Place it into `dataset` directory in the parent directory.

3. Install imblearn.
```bash
pip install imblearn
pip install wordcloud
```

4. Run the analysis part
```bash
python analysis.py
```

5. extract features from data
```bash
python main.py --do_data
```
6. Train the classifier
```bash
python main.py -classifier <classifier version (naive_bayes, logistic_regression, xgboost)>
```

e.g. to train `naive_bayes` classifier
```bash
python main.py -classifier naive_bayes
```

# Feature Extractor Details
## Set of Features
1. **TfIdf on shared words**
For each question pair,
- Stop words are removed.
- Shared words are extracted.
- TfIdf feature is extracted from the shared words only.
- This allows not to be biased to the common words that are more likely to be shared, because of their commonalities.
![image](https://drive.google.com/uc?export=view&id=1gx48hlaADSsPx-Jl9MjQmbCSWgWPZaz8)

2. **Word Match Share**
For each question pair,
- Stop words are removed.
- Shared words are extracted.
- Ratio of shared words is calculated, `R = (2 * number_of_shared_words) / (number_of_words_in_question1 + number_of_words_in_question2)`.
![image](https://drive.google.com/uc?export=view&id=127Xys73ZdB0SWhokczYCEol0GxJThqGz)

3. **Jaccard**
For each question pair,
- Stop words are removed.
- Shared words are extracted (intersection).
- All set of words in both questions are extracted (union).
- Ratio of shared words is calculated, `R = number_of_shared_words / number_of_union_words`.
![image](https://drive.google.com/uc?export=view&id=1lLO5KXfCTbNX4XbHeAUWGqjBMrZcfhIn)

4. **Word Count Difference**
For each question pair, absolute difference between number of words in questions is calculated.
![image](https://drive.google.com/uc?export=view&id=1pNv1I1ps4ATnFpmFe6-S2iwmSQWNIU5j)

5. **Word Count Ratio**
For each question pair,
- Word Count of both questions are calculated.
- Ratio of Counts is calculated, `R = min_word_count / max_word_count`.
![image](https://drive.google.com/uc?export=view&id=15sLwKbHN-cfSo_NfX2OaMCsH8XIpZuJW)

6. **Unique Word Count Difference**
For each question pair, absolute difference between number of unique words in questions is calculated.
![image](https://drive.google.com/uc?export=view&id=11e6IIEPH6L3cJDF7-banCdtwSGy_uUc9)

7. **Unique Word Count Difference without Stop words**
For each question pair, 
- Stop words are removed.
- Absolute difference between number of unique words in questions is calculated.
![image](https://drive.google.com/uc?export=view&id=13E00J3Lt7DjEAhgkgHbia00vQH29u39v)

8. **Word Match Count**
For each question pair, number of shared words is calculated.
![image](https://drive.google.com/uc?export=view&id=1dFLtL3fejd9ND3QQLAH4p1dXdap02Bpy)

9. **Unique Word Count**
For each question pair, number of unique words from both questions is calculated.
![image](https://drive.google.com/uc?export=view&id=1K02a3R0tYjEEVSKLYcOJjOZVnqIP8wek)

10. **Unique Word Count Ratio**
For each question pair,
- Unique Word Count of both questions are calculated.
- Ratio of Counts is calculated, `R = min_word_count / max_word_count`.
![image](https://drive.google.com/uc?export=view&id=1fL5fvuGzKmP6SewcG2I1ImEtHSsrpqV7)

11. **Unique Word Count Ratio without Stop words**
For each question pair,
- Stop words are removed.
- Unique Word Count of both questions are calculated.
- Ratio of Counts is calculated, `R = min_word_count / max_word_count`.
![image](https://drive.google.com/uc?export=view&id=1sP51ncq1yzsM43Z22YyI9VyDPXnWA2TX)

12. **Same Start Word**
For each question pair, check whether both questions start with the same word or not.
![image](https://drive.google.com/uc?export=view&id=19txbI6LzkOfGT0L89clMJ8e4xh7Xfzzo)

13. **Character Count Difference**
For each question pair, absolute difference between number of characters in questions is calculated.
![image](https://drive.google.com/uc?export=view&id=1YpkILB_40RzS1TMCQzvbIq5tbc7jdm8i)

14. **Character Count Ratio**
For each question pair,
- Character Count of both questions are calculated.
- Ratio of Counts is calculated, `R = min_character_count / max_character_count`
![image](https://drive.google.com/uc?export=view&id=1HhknoHcix9hwSqXwqQe-Z8P2VJReP4pv)


## Feature Selection
We applied a `Sequential Backward Selection` Approach to select the best representative features, and it ended-up with these features:
- Word Match Share
- Word Count Difference
- Word Count Difference without Stop words
- Unique Word Count Difference
- Unique Word Count Difference without Stop words
- Unique Word Count
- Unique Word Count Ratio
- Same Start Word
- Character Count Difference
- Character Count Ratio


## Classifiers
We splitted the dataset as `90% training` and `10% validation`, and here are the results
1. **Naive Bayes**
- Training Accuracy: 59%
- Validation Accuracy: 58.8%
- Validation AUC: 0.612
- ROC Curve:

![image](https://drive.google.com/uc?export=view&id=1qTjlJ4mFZwLCnEX12Uj5NetJRR8NjkLk)

2. **Logistic Regression**
- Training Accuracy: 70.27%
- Validation Accuracy: 70.46%
- Validation AUC: 0.783
- ROC Curve:

![image](https://drive.google.com/uc?export=view&id=1AKCw3SLTTqNAPcdxORVm7MdxpzEWVvRM)

3. **XGBoost**
- Training Accuracy: 78.26%
- Validation Accuracy: 78.32%
- Validation AUC: 0.874
- ROC Curve:

![image](https://drive.google.com/uc?export=view&id=19l2w2mZ3k2UcLD9heGJLgRPq9kRQ-uAM)

## Notes on Classical Approaches
- XGBoost has the best performance, while Naive Bayes has the least.
- It's not a linearly-separable problem as shown in features plots and higher performance in non-linear classifiers, such as XGBoost.
- Still not a very good performance, because this a complex problem that may need Advanced Deep Learning Techniques. This is what is said bu Qoura itself in the competition description:
`In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not.`

# Question Matching
This Module is designed to 
- get the most 5 matching questions from the dataset to the user's question.
- provide the user with a numerical value indicating how trendy his question is based on the number of similar questions.

### Here are some results:
1. Sample 1
- input question: `what are the names of the states of america?`
- Results: 
Most 5 matching questions:
```
What are the most common Last names in America?
Who was the all time best President of the United States of America?
What are the requirements to become president of the United States of America?
Why do you want to become the president of the United States of America?
What are some of the greatest achievements of the United States of America?
```
Trend Value: `125`

2. Sample 2
- input question: `what is the definition of quantum entanglement?`
- Results: 
Most 5 matching questions:
```
What is quantum entanglement? What is mathematics behind quantum entanglement?
Is it possible to transmit power wirelessly through quantum entanglement?
What is the mathematics behind quantum entanglement?
What is quantum entanglement?
What are the proofs of quantum entanglement?
```
Trend Value: `263`

3. Sample 3
- input question: `is it a new world war?`
- Results: 
Most 5 matching questions:
```
How close are we to world war?
Should the world be worried about the new "Islamic State" caliphate?
Will there be a third World War?
Will there be another big World War? If so, what will it be like?
What would the world do if India and Pakistan go to war?
```
Trend Value: `272`



