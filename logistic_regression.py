import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--path", required = True, help = "path to the dataset csv file")
args = parser.parse_args()


df = pd.read_csv(args.path)          # read in CSV
df['text'] = df['text'].fillna('')   # gets rid of null characters
df = df.sample(frac=1, random_state=random.randint(0,255))  # randomizes order of data
text = df['text']
labels = df['label']

# splits the text along with respective labels into sections. 80% is for training, 20% is for testing.
state = random.randint(0, 255)    # state is just random seed for splitting
train_text, test_text, train_labels, test_labels = train_test_split(text, labels, test_size=.2, random_state=state)

# vectorizes the training and testing texts.
vectorizer = TfidfVectorizer(max_features=9000, min_df=5)     
transformed_text = vectorizer.fit_transform(train_text)
transformed_test_text = vectorizer.transform(test_text)

# trains the model, uses l1 regularization with liblinear solver since the datasets are relatively small
log_reg_model = LogisticRegression(penalty='l1', C=1, solver='liblinear')
log_reg_model.fit(transformed_text, train_labels)

# predicts labels and gets the total accuracy of them across test set
predicted_labels = log_reg_model.predict(transformed_test_text)
accuracy_test = accuracy_score(test_labels, predicted_labels)

print("Testing Accuracy:", accuracy_test)