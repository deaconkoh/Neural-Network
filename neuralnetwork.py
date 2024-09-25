import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


unlabeled_df = pd.read_csv("emails.csv")

def extract_email_contents(data):
    head, body = data.split("\n\n", 1)
    
    feature = {}
    
    #extract important contents of the header
    feature["body"] = body.strip()
    feature["from"] = re.search(r"From: (.+)", head).group(1) if re.search(r"From: (.+)",head) else ""
    feature["domain"] = feature["from"].split('@')[-1] if '@' in feature["from"] else feature["from"]
    feature["subject"] = re.search(r"Subject: (.+)", head).group(1) if re.search(r"Subject: (.+)",head) else ""

    #check if the body indicates if there is any attachment
    temp_attachment = re.findall(r"Content-Disposition: attachment(?:;.*filename=\"(.+)\")?", data)
    feature["attachment"] = ", ".join(temp_attachment) if temp_attachment else ""
    
    return feature


#creating a new dataframe extracted_data containing all the email features
extracted_data = unlabeled_df["message"].apply(extract_email_contents).apply(pd.Series)

"""
Encoding the attachment column to either a 0 (dont have) or 1 (Have) and clustering them together
"""
#encoding the attachment & links portion to either 0(no attachements) or 1(have attachements)
extracted_data["has_attachment"] = extracted_data["attachment"].apply(lambda x: 1 if x!="" else 0)
binary_clusters = extracted_data["has_attachment"].values

"""
Using Frequency encoding to encode the domain columns and clustering them together
"""
#encode the domain using frequency encoding 
#For each unique category, count how many times it appears in the dataset.
#Replace the category with the count or frequency value.
domain_freq = extracted_data["domain"].value_counts(normalize=True)
extracted_data["domain_freq"] = extracted_data["domain"].map(domain_freq)

# Cluster the frequency-encoded 'domain' feature
kmeans_domain = KMeans(n_clusters=2, random_state=42)
domain_clusters = kmeans_domain.fit_predict(extracted_data[["domain_freq"]])


"""
Vectorizing text contents using TF-IDF and clustering them using kmeans
"""
#vectorizing contents in the dataframe
vectorize = TfidfVectorizer(max_features=1000)
extracted_data["combined_text"] = extracted_data["body"] + " " + extracted_data["subject"] + " " + extracted_data["from"]
x_text = vectorize.fit_transform(extracted_data["combined_text"]).toarray()

# Cluster the TF-IDF vectors from the text data
kmeans_text = KMeans(n_clusters=2, random_state=42)
text_clusters = kmeans_text.fit_predict(x_text)

""""
Creating the multi input layers for Neural Network
"""
#Creating the different input layer
input_text = Input(shape=(x_text.shape[1],), name="text_input")
input_domain = Input(shape=(1,), name='domain_input')
input_attachment = Input(shape=(1,), name='attachment_input')

#text feature processing
text_dense = Dense(128, activation='relu')(input_text)

# Domain frequency processing
domain_dense = Dense(10, activation='relu')(input_domain)

# Attachment binary feature processing
attachment_dense = Dense(10, activation='relu')(input_attachment)

# Concatenate the processed inputs
combined = Concatenate()([text_dense, domain_dense, attachment_dense])

# Add more dense layers after concatenation
dense_1 = Dense(64, activation='relu')(combined)
output = Dense(1, activation='sigmoid')(dense_1)  # Binary classification (e.g., phishing vs. non-phishing)

# Define the multi-input model
model = Model(inputs=[input_text, input_domain, input_attachment], outputs=output)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


pseudo_labels = text_clusters  # Use the text clusters as the labels

# Preparing the inputs for the neural network
X_text = x_text  # TF-IDF vectorized text data
X_domain = extracted_data["domain_freq"].values.reshape(-1, 1)  # Domain frequency as input
X_attachment = extracted_data["has_attachment"].values.reshape(-1, 1)  # Binary attachment
from sklearn.model_selection import train_test_split

# Split your data into training and test sets
X_text_train, X_text_test, X_domain_train, X_domain_test, X_attachment_train, X_attachment_test, y_train, y_test = train_test_split(
    X_text, X_domain, X_attachment, pseudo_labels, test_size=0.2, random_state=42)

# Train the model on the training data
history = model.fit([X_text_train, X_domain_train, X_attachment_train], y_train, epochs=10, batch_size=32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate([X_text_test, X_domain_test, X_attachment_test], y_test)
