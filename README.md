# Introduction

This project is about a profile-based document filtering. The goal is to manage a incoming flow of documents, and to assignment each document to the users with a corresponding profile. It can be very useful for news website or mobile application. Indeed, if the users choose the main fields they are interested in, the website (or the app) can process automatically the latest articles to recommend them to the users that would be interested in.

For this work, we will have 5 users (Thomas, Aline, Georges, Eva and Lorenzo) with different interests from this list: Cars, Sports, Science, Religion, Politics.


# The method used

For this project, I wanted to have to possibilty for the user to paste the text he wants to classify. The output must be the main theme of the text and the users it will be delivered to. 

We will need to have a classifier to predict the main theme of the text.  We will use Natural Language Processing tools to preprocess the text and vectorize.  
We will need a training dataset to train our classifier.  
This work will be coded in Python.  
  

## The dataset

I used the dataset `fetch_20newsgroups` from the `sklearn` framework. This is a dataset with 18000 newsgroups posts about 20 differents topics: `'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'`. I regrouped these topics in 5 topics (and I didn't keep the `comp.*` topics): `'cars', 'sport', 'science', 'religion', 'politics'`.  The dataset can be split in two subsets: one for training and the other one for testing.  
This will be the dataset we use to train our article classifier.

To test it and use it, we can use articles from the BBC News Website for instance. (https://www.bbc.com/news)


## The preprocessing of the text

We want our text to be the input of a classifier. We need to preprocess it, we can't just give a string to our classifier. We have transform our text into another type of representation.

We will transform our text into a vector, with the **TF-IDF calculation**. TF-IDF stands for **Term Frequency - Inverse Document Frequency**. This is a measure of the originality of a words using the number of time a words appears in a document (Term Frequency) and the number of different documents in whiwh the word appears in (Document Frequency).  
![equation](http://www.sciweavers.org/tex2img.php?eq=%20tfidf%28%20t%2C%20d%2C%20D%20%29%20%3D%20tf%28%20t%2C%20d%20%29%20%5Ctimes%20idf%28%20t%2C%20D%20%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

#### The Term Frequency
is easy to understand. It calculates the frequence at which the word t appears in the document d. Usually we have:  
![img](http://www.sciweavers.org/tex2img.php?eq=tf%28%20t%2C%20d%20%29%20%3D%20%5Clog%20%28%201%20%2B%20%5Cfrac%7B%5Ctext%7Btotal%20documents%20in%20corpus%7D%7D%7B%5Ctext%7Bdocuments%20with%20term%7D%7D%29&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0)

#### The Inverse document Frequency
is a way to calculate how xommon or rare a word is in the entire corpus. It is the *inverse* document frequency so the closer it is to 0, the most common a word is (and vice versa). This is how we calculated it:  
![img](http://www.sciweavers.org/tex2img.php?eq=idf%28%20t%2C%20D%20%29%20%3D%20log%20%5Cfrac%7B%20%5Ctext%7B%7C%20%7D%20D%20%5Ctext%7B%20%7C%7D%20%7D%7B%201%20%2B%20%5Ctext%7B%7C%7D%20%5C%7B%20d%20%5Cin%20D%20%3A%20t%20%5Cin%20d%20%5C%7D%20%5Ctext%7B%7C%7D%20%7D&bc=White&fc=Black&im=png&fs=12&ff=mathdesign&edit=0)



So, the **TF-IDF** is an accurate way to calculate how important a word is in a document, and how reliable is it to predict the theme of the document. The closer is it to 1, the more important the word is.

Before doing this, we need to have homogenous data. Their is an important step before TF-IDF calculation which is the preparation of the text.

#### Preparation of the text  
**Tokenization:** The text has to be split into sentences and the sentences into words. The words must be lowercased, and the punctuation removed.  
**Stemming:** The words are reduced to their root form (e.g.: eating $\rightarrow$ eat).  
**Lemmatization:** The words in third person are changed to first person, verbs in past or future tenses are changed into present.  
The **stopwords** are removed. It's the list of the most commonly used words (e.g.: 'the', 'and', 'a').  
We removed the words **shorter than 3 characters**.  


## The classifier

Now, for each text, we have a vector representing it. We will use this vector as an input for our classifier. The classifier I chose is a Support Vector Machine, because we have a high dimensionality input vector, and not that much of training samples.  
First, we will have to train that SVM thanks to our training dataset.
Then, we will predict the article input by the user of our programm.  

Let's see what is my implementation for this project. 



# The implementation

To see the full implementation of my work, visit this github page: 

## The dataset

First, we have to download the dataset and split it into a training and testing dataset. Before this, I chose the categories I want to keep in the dataset (from the list in part 2.1).


```python
cats = ['rec.autos', 
        'rec.sport.baseball', 
        'rec.sport.hockey', 
        'sci.crypt', 
        'sci.electronics', 
        'sci.med', 
        'sci.space', 
        'soc.religion.christian', 
        'talk.politics.guns', 
        'talk.politics.mideast', 
        'talk.politics.misc', 
        'talk.religion.misc']
train_dataset = fetch_20newsgroups(subset='train', categories=cats, shuffle=True)
test_dataset = fetch_20newsgroups(subset='test', categories=cats, shuffle=True)
```

## The preprocessing of the data

#### The preparation of the data  
Then, as I said before, we have to define function to prepare our text. We will use these functions to prepare the training and testing dataset, but also to prepare the input of the user (the text or article he wants my programm to classify).  
For this, we use a Stemmer and a Lemmatizer from the `nltk` Python library. We also use `STOPWORDS` from the `gensim` library.  
After this preparation, we get the text only with the root of the words, longer than 3 characters, that are not in the stop-words list.


```python
#Tokenize and lemmatize a text

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def stem_and_lemmatize(text):
    lemm_text = lemmatizer.lemmatize(text, pos='v')
    return stemmer.stem(lemm_text)

def preprocess_data(text):
    """
    This function returns the text preprocessed: a new text with the words stemmed and lemmatized, without the stop words and the words shorter than 3 letters.
    """
    words = []
    text = simple_preprocess(text) #the text is converted into a list of lower-case words
    for token in text:
        if token not in STOPWORDS and len(token) > 3:
            words.append(stem_and_lemmatize(token))
    return ' '.join(words)
```

#### The TF-IDF calculation  
The next function will end the preprocessing of the training and testig dataset by returning the list of TF-IDF vectors for each text of both dataset.


```python
def preprocess_dataset(train_dataset, test_dataset):
    """
    This function preprocess the dataset.
    It returns:
        The Tfidf vectors of training and testing datasets (X_train, X_test)
        The labels (y_train, y_test)
        The Tfidf vectorizer object
    """
    train_processed_texts = []
    for text in train_dataset.data:
        train_processed_texts.append(preprocess_data(text))
    test_processed_texts = []
    for text in test_dataset.data:
        test_processed_texts.append(preprocess_data(text))

    vect = TfidfVectorizer(stop_words='english', min_df=2)
    X_train = vect.fit_transform(train_processed_texts)
    X_test = vect.transform(test_processed_texts)
    y_train = np.array(train_dataset.target)
    y_test = np.array(test_dataset.target)

    return X_train, X_test, y_train, y_test, vect

X_train, X_test, y_train, y_test, vect = preprocess_dataset(train_dataset, test_dataset)
```

To calculate the TF-IDF of texts from a corpus, we use a `TfidfVectorizer` from the `sklearn` framework. After we have fit the training dataset in this object, we will be able to use it in order to convert the input article into a vector, as we will see later.

## The classifier

I chose a SVM classifier to classify our texts.  
First, we need to transform the labels of the 20newsgroups dataset, so that it corresponds to the categories we chose.


```python
# Create a dictionarry to associate the labels of the dataset with name of categories
dict_cats = {'cars': [0], 'sport': [1, 2], 'science': [3, 4, 5, 6], 'religion': [7, 11], 'politics': [8, 9, 10]}

# Create a dictionarry to associate the labels of the dataset with new numbers of categories
dict_cats_number = {0: [0], 1: [1, 2], 2: [3, 4, 5, 6], 3: [7, 11], 4: [8, 9, 10]}


def transform_label(labels):
    labels2 = [0 for k in range(len(labels))]
    for i in range(len(labels)):
        label = labels[i]
        for j in range(5):
            if label in dict_cats_number[j]:
                labels2[i] = j
                break
    return labels2
```

Then, we will train our SVM. This means that it will finds the best hyperplans to differentiates the differents classes. I use a linear kernel here because we have a high number of features, so it is very likely that the data will be linearly separable.


```python
def trained_svm(X_train, y_train, kernel):
    model = svm.SVC(kernel=kernel, gamma='auto')
    model.fit(X_train, y_train)
    return model
```

Now, we can use our testing dataset to evaluate the performance of our classifier.


```python
y_train = transform_label(y_train)
y_test = transform_label(y_test)

print("Training...")
kernel = 'linear'
SVM = trained_svm(X_train, y_train, kernel)
print("Training done.")

y_pred = SVM.predict(X_test)
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(report)
print("Accuracy: ", accuracy)
```

    
    Training...
    Training done.
                  precision    recall  f1-score   support
    
               0       1.00      0.76      0.86       396
               1       0.98      1.00      0.99      4074
    
        accuracy                           0.98      4470
       macro avg       0.99      0.88      0.92      4470
    weighted avg       0.98      0.98      0.98      4470
    
    Accuracy:  0.9782997762863535
    
    

We have a pretty good accuracy, so we will use this SVM for the final programm of this project. We have to remember that we want that the user paste a text or an article of his choice, and that our programm deliver this text to one of the user we defined.

# The main programm

#### Definition of the profiles  
First, we have to define the profiles we want to use in our programm. Here, I used a dictionnary: the keys are the name of the users and the values are the theme there are interested in.


```python
profiles = {
    'Thomas' : ['sport', 'politics'],
    'Aline' : ['cars'],
    'George' : ['religion', 'cars'],
    'Eva' : ['science', 'politics'],
    'Lorenzo' : ['sport']
}
```

Then, I defined a function that will return the predicted category of the text. It uses first the `TfidfVectorizer` object (`vect`) to transform the text into the right vector for our classifier. Then, our SVM will predict the category of the text (it will give a number between 0 and 4 and this number is converted into the name of the category: sport, politics, ...).


```python
def predict_text(text):
    """
    This function returns the predicted category of an input text.
    """
    text = [preprocess_data(text)]
    text = vect.transform(text)
    prediction = SVM.predict(text)[0]
    cat = list(dict_cats.keys())[prediction]
    return cat
```

The function `main()` is the function to create the very little user interface: the user will paste his text in an input field, and the programm will print the category of the text, and to which users it must be delivered.  
If you want to run this programm, run the notebook 'code.ipynb' in the github repository of this project.


```python
def main():
        text = input("Please type the text you want and press enter: ")
        print("Processing...")
        category = predict_text(text)
        users = []
        for user in profiles.keys():
            if category in profiles[user]:
                users.append(user)
        print("The theme of this text is: ", category)
        print('')
        print("So, this text will be sent to the following users:")
        for user in users:
            print("     -  ", user)
```

You can try by pasting articles from the BBC website. Choose article from these categories: science, cars, politics, sports, religion.


# Conclusion

We have coded a simple profile-based article classifier. It uses Natural Language Processing to retrieve information from a text and classify it. I could improve my project by having more categories such as cinema, music, etc. I could also have a more complete dataset. For instance, this dataset has only two sports: hockey and baseball. So, sometimes it doesn't work when the article is about other sports. We also could have use other NLP techniques to represent our texts (such as word2vec).  
But this classifier is working well for a simple using with news article.
