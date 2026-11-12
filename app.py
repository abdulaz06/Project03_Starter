"""
UIC CS 411 - Project 03: Naive Bayes Text Classifier from Scratch
"""

import csv
import math
import re


word_counts = {"positive": {}, "negative": {}}
total_words = {"positive": 0, "negative": 0}
class_counts = {"positive": 0, "negative": 0}
class_priors = {}
vocabulary = set()


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def train_model(train_file="data/train.csv"):
    global word_counts, total_words, class_counts, class_priors, vocabulary

    word_counts = {"positive": {}, "negative": {}}
    total_words = {"positive": 0, "negative": 0}
    class_counts = {"positive": 0, "negative": 0}
    class_priors = {}
    vocabulary = set()

    with open(train_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row["label"].strip()
            text = row["text"]

            if label not in class_counts:
                raise ValueError("Invalid label: " + label)

            class_counts[label] += 1

            for word in tokenize(text):
                vocabulary.add(word)
                word_counts[label][word] = word_counts[label].get(word, 0) + 1
                total_words[label] += 1

    total_sentences = sum(class_counts.values())
    if class_counts["positive"] == 0 or class_counts["negative"] == 0:
        raise ValueError("The training file needs both positive and negative sentences.")

    for label in class_counts:
        class_priors[label] = class_counts[label] / total_sentences


def calculate_score(text, label):
    if not class_priors:
        raise ValueError("Call train_model() before predicting.")

    score = math.log(class_priors[label])
    denominator = total_words[label] + len(vocabulary)

    for word in tokenize(text):
        if word in vocabulary:
            count = word_counts[label].get(word, 0)
            probability = (count + 1) / denominator
            score += math.log(probability)

    return score


def predict(text):
    positive_score = calculate_score(text, "positive")
    negative_score = calculate_score(text, "negative")

    if positive_score > negative_score:
        return "positive"
    if negative_score > positive_score:
        return "negative"

    if class_priors["negative"] > class_priors["positive"]:
        return "negative"
    return "positive"


def evaluate_model(test_file="data/test.csv"):
    confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    correct = 0
    total = 0

    with open(test_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            actual = row["label"].strip()
            predicted = predict(row["text"])

            if actual not in class_counts:
                raise ValueError("Invalid label: " + actual)

            total += 1
            if actual == predicted:
                correct += 1

            if actual == "positive" and predicted == "positive":
                confusion_matrix["TP"] += 1
            elif actual == "negative" and predicted == "negative":
                confusion_matrix["TN"] += 1
            elif actual == "negative" and predicted == "positive":
                confusion_matrix["FP"] += 1
            else:
                confusion_matrix["FN"] += 1

    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "confusion_matrix": confusion_matrix}


if __name__ == "__main__":
    train_model()
    print("Vocabulary size:", len(vocabulary))
    print("Positive sentences:", class_counts["positive"])
    print("Negative sentences:", class_counts["negative"])
    print("Positive word count:", total_words["positive"])
    print("Negative word count:", total_words["negative"])
    print("Class priors:", class_priors)

    results = evaluate_model()
    print("\nTest accuracy:", round(results["accuracy"] * 100, 2), "%")
    print("Confusion matrix:", results["confusion_matrix"])

    while True:
        try:
            sentence = input("\nEnter a sentence (or 'quit'): ")
        except (EOFError, KeyboardInterrupt):
            break

        if sentence.strip().lower() == "quit" or not sentence.strip():
            break

        print("Tokens:", tokenize(sentence))
        print("Positive log score:", round(calculate_score(sentence, "positive"), 4))
        print("Negative log score:", round(calculate_score(sentence, "negative"), 4))
        print("Prediction:", predict(sentence))
