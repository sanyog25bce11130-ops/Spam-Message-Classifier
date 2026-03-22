import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# 1. Loading dataset (using the 'Data' folder you just created)
# We use '../Data/Spam.csv' because the script is inside the 'src' folder
data_path = os.path.join('Data', 'Spam.csv')
data = pd.read_csv(data_path)

# 2. Separating input (messages) and output (labels)
X = data['message']
y = data['label']

# 3. Converting text into numbers
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

# 4. Training the model
model = MultinomialNB()
model.fit(X_vector, y)

# 5. Saving the model and vectorizer so we can use them later
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model trained and saved successfully!")
