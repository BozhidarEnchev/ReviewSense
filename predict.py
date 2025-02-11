from review_sense import ReviewSense
review_sense = ReviewSense()

while True:
    text = input("enter text: ")
    prediction = review_sense.predict_sentiment(text)
    print(prediction)
