# Resume-Screening-Using-NLP

This repository contains Python scripts for analyzing a resume dataset. The dataset consists of resumes in a CSV format, and the scripts perform various preprocessing and analysis tasks on the text data.

### Code 1

The `data_model_building.py` script focuses on data preprocessing and exploratory data analysis. It performs the following tasks:

1. **Import Dependencies**: The necessary libraries and modules are imported.

2. **Load Dataset**: The resume data is loaded from a CSV file into a pandas DataFrame.

3. **Exploratory Data Analysis**: Basic exploratory analysis is conducted on the dataset, including checking the dataset's shape, identifying missing values and duplicates, and providing an overview of the dataset's columns.

4. **Text Preprocessing**: The script analyzes the resume text data, calculating various statistics such as word count, character count, average word length, stopwords count, numerics count, and uppercase words count. Histograms are used to visualize the results.

5. **Lemmatization & Stemming**: Text preprocessing techniques such as lemmatization and stemming are applied to the resume text. This involves removing stopwords, punctuation, and converting the text to lowercase. Tokenization, stemming, and lemmatization are then performed on the text.

6. **Text Visualization using Wordcloud**: Word clouds are generated to visualize the most common words in the resumes. Both an overall word cloud for the dataset and individual word clouds for each job category are created.

7. **N-grams**: The script extracts unigrams and bigrams from the resume text using the ngrams function from the TextBlob library. Examples of unigrams, bigrams, and trigrams are provided.

8. **Count Vectorizer on N-grams**: The CountVectorizer from scikit-learn is used to create a matrix of ngrams (unigrams and bigrams) and count the frequency of each ngram. A bar chart is generated to display the top 20 most used words in the resumes.

9. **Named Entity Recognition (NER)**: Named entity recognition is applied to the preprocessed resume text using the spaCy library. The script identifies and visualizes named entities such as organizations, persons, locations, etc.

### Code 2

The `data_preprocessing.py` script also focuses on data preprocessing and analysis of the resume dataset. It performs similar tasks as Code 1, but with some variations. The tasks include:

1. **Import Dependencies**: The necessary libraries and modules are imported.

2. **Load Dataset**: The resume data is loaded from a CSV file into a pandas DataFrame.

3. **Exploratory Data Analysis**: Basic exploratory analysis is conducted on the dataset, checking its shape, identifying missing values and duplicates, and providing an overview of the dataset's columns.

4. **Text Preprocessing**: The script analyzes the resume text data, calculating various statistics such as word count, character count, average word length, stopwords count, numerics count, and uppercase words count. Histograms are used to visualize the results.

5. **Lemmatization & Stemming**: Text preprocessing techniques such as lemmatization and stemming are applied to the resume text. This involves removing stopwords, punctuation, and converting the text to lowercase. Tokenization, stemming, and lemmatization are then performed on the text.

6. **Text Visualization using Wordcloud**: Word clouds are generated to visualize the most common words in the resumes. Both an overall word cloud for the dataset and individual word clouds for each job category are created.

7. **N-grams**: The script extracts unigrams and bigrams from the resume text using the ngrams function from the TextBlob library.

8. **Count Vectorizer on N-grams**: The CountVectorizer from scikit-learn is used

 to create a matrix of ngrams (unigrams and bigrams) and count the frequency of each ngram. A bar chart is generated to display the top 20 most used words in the resumes.

These scripts provide valuable insights into the resume dataset, enabling data preprocessing, exploratory analysis, and visualization of text data. They are helpful for understanding the characteristics of the resumes and extracting meaningful information from the text.
