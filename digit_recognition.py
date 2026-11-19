# Handwritten Digit Recognition using Machine Learning
# Made by: Soumil Malik (26/B06/038) and Sukhman Singh (26/B06/041)
# Goal: Look at a picture of a handwritten digit and tell which number (0-9) it is

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------- Step 1: Load the dataset ----------------
digits = load_digits()
X = digits.data          # each row = one image (64 pixel values)
y = digits.target        # the correct digit for each image (0 to 9)

print("Total images:", len(X))
print("Pixels in each image:", len(X[0]))
print()
print("Pixel values of the first image (8 x 8 grid):")
print(digits.images[0])
print("This image is the digit:", y[0])
print()

# ---------------- Step 2: Understand the data ----------------
# Graph 1: show the first 10 images with their correct digit
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap="gray_r")
    plt.title("Digit: " + str(y[i]))
    plt.axis("off")
plt.suptitle("Sample Images from the Dataset")
plt.savefig("graph1_sample_digits.png")
plt.close()

# Count how many images there are of each digit
counts = pd.Series(y).value_counts().sort_index()
print("Number of images of each digit:")
print(counts)
print()

# Graph 2: bar chart of the count of each digit
plt.bar(counts.index, counts.values, color="teal")
plt.xticks(range(10))
plt.title("Number of Images of Each Digit")
plt.xlabel("Digit")
plt.ylabel("Number of Images")
plt.savefig("graph2_digit_count.png")
plt.close()

# ---------------- Step 3: Split data into training and testing ----------------
# Pixel values are from 0 to 16, so we divide by 16 to bring them between 0 and 1
X = X / 16

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training images:", len(X_train))
print("Testing images:", len(X_test))
print()

# Graph 3: show 40 images from the training data that the models learn from
plt.figure(figsize=(10, 5))
for i in range(40):
    plt.subplot(4, 10, i + 1)
    plt.imshow(X_train[i].reshape(8, 8), cmap="gray_r")
    plt.title(str(y_train[i]), fontsize=9)
    plt.axis("off")
plt.suptitle("40 Sample Images from the Training Data")
plt.savefig("graph3_training_samples.png")
plt.close()

# ---------------- Step 4: Train three models ----------------
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
acc1 = accuracy_score(y_test, model1.predict(X_test))

model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X_train, y_train)
acc2 = accuracy_score(y_test, model2.predict(X_test))

model3 = DecisionTreeClassifier(random_state=42)
model3.fit(X_train, y_train)
acc3 = accuracy_score(y_test, model3.predict(X_test))

print("Accuracy of Logistic Regression:", round(acc1 * 100, 2), "%")
print("Accuracy of K-Nearest Neighbours:", round(acc2 * 100, 2), "%")
print("Accuracy of Decision Tree:", round(acc3 * 100, 2), "%")
print()

# Graph 4: compare accuracy of the models
names = ["Logistic\nRegression", "KNN", "Decision\nTree"]
scores = [acc1 * 100, acc2 * 100, acc3 * 100]
plt.bar(names, scores, color=["blue", "orange", "purple"])
plt.ylim(70, 100)
plt.title("Accuracy of Different Models")
plt.ylabel("Accuracy (%)")
plt.savefig("graph4_model_accuracy.png")
plt.close()

# ---------------- Step 5: Check the best model in detail ----------------
# KNN gave the highest accuracy, so we use it
predictions = model2.predict(X_test)
wrong = (predictions != y_test).sum()
print("Wrong predictions by KNN:", wrong, "out of", len(y_test))
print()

# Graph 5: confusion matrix (rows = actual digit, columns = predicted digit)
cm = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title("Confusion Matrix - KNN")
plt.savefig("graph5_confusion_matrix.png")
plt.close()

# ---------------- Step 6: Predict some test images ----------------
# Graph 6: show 10 test images with the model's prediction
for i in range(10):
    image = X_test[i].reshape(8, 8)
    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap="gray_r")
    plt.title("Predicted: " + str(predictions[i]), fontsize=9)
    plt.axis("off")
plt.suptitle("Predictions on Test Images")
plt.savefig("graph6_predictions.png")
plt.close()

print("First 10 test images:")
print("Actual digits:   ", y_test[:10])
print("Predicted digits:", predictions[:10])
