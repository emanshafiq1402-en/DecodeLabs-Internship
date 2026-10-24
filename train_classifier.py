# Project 2: Data Classification Using AI (Iris + KNN)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# 1. LOAD DATA
iris = load_iris()
X, y = iris.data, iris.target
print("Samples:", X.shape[0], "| Features:", X.shape[1])
print("Features:", iris.feature_names)
print("Classes:", list(iris.target_names))

# 2. SPLIT (80% train, 20% test, shuffled)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)
print("\nTrain size:", len(X_train), "| Test size:", len(X_test))

# 3. SCALE (fit on train only, then apply to test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. FIND BEST K
print("\nTesting different K values:")
best_k, best_score = 1, 0
for k in range(1, 16):
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    score = accuracy_score(y_test, m.predict(X_test))
    print(f"  K={k:2d} -> accuracy {score:.3f}")
    if score > best_score:
        best_k, best_score = k, score
print("Best K:", best_k)

# 5. TRAIN FINAL MODEL (K=5 as in the slides)
model = KNeighborsClassifier(n_neighbors=5)   # instantiate
model.fit(X_train, y_train)                   # fit
predictions = model.predict(X_test)           # predict

# 6. EVALUATE
print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("F1 Score (weighted):", round(f1_score(y_test, predictions, average="weighted"), 3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# 7. PREDICT A NEW FLOWER
new_flower = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
print("New flower prediction:", iris.target_names[model.predict(new_flower)[0]])

# PICTURES
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Picture 1: Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=iris.target_names
)
plt.title("Confusion Matrix")
plt.show()

# Picture 2: Error rate for different K values
errors = []
for k in range(1, 16):
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    errors.append(1 - accuracy_score(y_test, m.predict(X_test)))

plt.plot(range(1, 16), errors, marker="o")
plt.xlabel("K value")
plt.ylabel("Error rate")
plt.title("Choosing K")
plt.show()