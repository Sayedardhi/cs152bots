from transformers import pipeline


classifier = pipeline(
    "text-classification",
    model="ekurtulus/cyberbullying_classifier",
)

def classify_message(message):
    """
    Classifies a message using the cyberbullying classifier.
    Returns a tuple of (label, score) where label is the classification and score is the confidence.
    """
    result = classifier(message)[0]
    return result['label'], result['score']

print(classify_message("You should kill yourself!"))
