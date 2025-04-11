import matplotlib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class ReviewSense:
    """
    A class to train a sentiment analysis model on IMDB movie reviews.
    It uses an LSTM model to predict sentiment (positive/negative) based on the review text.
    """
    def __init__(self):
        self.max_words = 10000
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.max_length = 200
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.data = self.prepare_data()

    def train(self):
        """Train the model."""
        sequences = self.tokenizer.texts_to_sequences(self.data['review'])

        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')

        y = self.data['sentiment'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = Sequential([
            Embedding(input_dim=self.max_words, output_dim=128),  # Word embeddings
            LSTM(64, return_sequences=True),  # LSTM layer
            LSTM(32),  # Second LSTM layer
            Dropout(0.5),  # Reduce overfitting
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # Sigmoid for binary classification
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f'Test Accuracy: {test_acc:.4f}')

        y_pred_probs = self.model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

        self.model.save("sentiment_model.keras")

    def prepare_data(self):
        """Prepare the data for training"""
        data = pd.read_csv('data/IMDB Dataset.csv')

        data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

        data['review'] = data['review'].apply(self.clean_text)

        self.tokenizer.fit_on_texts(data['review'])

        return data

    def predict_sentiment(self, text):
        if not self.model:
            self.model = load_model("sentiment_model.keras")

        text = self.clean_text(text)
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_length, padding='post')
        prediction = self.model.predict(padded)[0][0]
        return "Positive" if prediction > 0.5 else "Negative"

    def clean_text(self, text):
        """Converts text to lowercase and removes stop words and punctuation"""
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        return text


def main():
    # Train the model with all data
    ReviewSense().train()


if __name__ == '__main__':
    main()
