import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from argparse import ArgumentParser
from sklearn import svm


parser = ArgumentParser()
parser.add_argument("--path", required = True, help = "path to the dataset csv file")
args = parser.parse_args()

df = pd.read_csv(args.path)          # read in CSV
df['text'] = df['text'].fillna('')   # get rid of null characters
text = df['text'].tolist()           # converts them to lists for clf
labels = df['label'].tolist() 

vectorizer = TfidfVectorizer(max_features=9000, min_df=5) # vectorizes text, also the parameters make it not select rarely used words   
text = vectorizer.fit_transform(text)                     # same with logistic_regression.py, testing was done and those nums work well

state = random.randint(0, 100)  # randomized seeds from 0 to 100 (arbitrary range)
train_text, test_text, train_labels, test_labels = train_test_split(text, labels, test_size=.2, random_state=state)

clf = svm.SVC(kernel='linear', C=1).fit(train_text, train_labels)
print("Testing Accuracy: ", clf.score(test_text, test_labels)) # print out the evaluated score