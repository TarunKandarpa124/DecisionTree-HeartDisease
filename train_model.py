import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt

def preprocess_data(data):
    # Drop rows with missing values
    data.dropna(inplace=True)

    # Standardize boolean values to 'TRUE' and 'FALSE'
    data = data.applymap(lambda x: 'TRUE' if x == 'TURE' else x)
    data = data.applymap(lambda x: 'FALSE' if x == 'FALSE' else x)

    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove outliers using IQR
    Q1 = data[numeric_columns].quantile(0.25)
    Q3 = data[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

    return data

# Load the training data
data = pd.read_excel('heart_disease.xlsx', sheet_name='Heart_disease')

# Preprocess the data
data = preprocess_data(data)

# Separate features and target
X = data.drop(['fbs'], axis=1)
y = data['fbs']

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# Convert boolean columns to strings
for col in categorical_columns:
    X[col] = X[col].astype(str)

# Preprocess the training data
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

encoders = {}
for col in categorical_columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    encoders[col] = encoder  # Save the encoder for each column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
tree = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}

grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

# Calculate evaluation metrics for training and test sets
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
train_precision = precision_score(y_train, y_pred_train)
test_precision = precision_score(y_test, y_pred_test)
train_recall = recall_score(y_train, y_pred_train)
test_recall = recall_score(y_test, y_pred_test)
train_f1 = f1_score(y_train, y_pred_train)
test_f1 = f1_score(y_test, y_pred_test)
train_roc_auc = roc_auc_score(y_train, y_pred_train)
test_roc_auc = roc_auc_score(y_test, y_pred_test)

# Print evaluation metrics
print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Train precision: {train_precision:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Train recall: {train_recall:.4f}")
print(f"Test recall: {test_recall:.4f}")
print(f"Train F1 score: {train_f1:.4f}")
print(f"Test F1 score: {test_f1:.4f}")
print(f"Train ROC AUC score: {train_roc_auc:.4f}")
print(f"Test ROC AUC score: {test_roc_auc:.4f}")

# Evaluate other tree models
tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_leaf=1, min_samples_split=2)
tree1.fit(X_train, y_train)

y_pred_train1 = tree1.predict(X_train)
y_pred_test1 = tree1.predict(X_test)

train_accuracy1 = accuracy_score(y_train, y_pred_train1)
test_accuracy1 = accuracy_score(y_test, y_pred_test1)
train_precision1 = precision_score(y_train, y_pred_train1)
test_precision1 = precision_score(y_test, y_pred_test1)
train_recall1 = recall_score(y_train, y_pred_train1)
test_recall1 = recall_score(y_test, y_pred_test1)
train_f11 = f1_score(y_train, y_pred_train1)
test_f11 = f1_score(y_test, y_pred_test1)
train_roc_auc1 = roc_auc_score(y_train, y_pred_train1)
test_roc_auc1 = roc_auc_score(y_test, y_pred_test1)

print(f"Tree1 Train accuracy: {train_accuracy1:.4f}")
print(f"Tree1 Test accuracy: {test_accuracy1:.4f}")
print(f"Tree1 Train precision: {train_precision1:.4f}")
print(f"Tree1 Test precision: {test_precision1:.4f}")
print(f"Tree1 Train recall: {train_recall1:.4f}")
print(f"Tree1 Test recall: {test_recall1:.4f}")
print(f"Tree1 Train F1 score: {train_f11:.4f}")
print(f"Tree1 Test F1 score: {test_f11:.4f}")
print(f"Tree1 Train ROC AUC score: {train_roc_auc1:.4f}")
print(f"Tree1 Test ROC AUC score: {test_roc_auc1:.4f}")

tree2 = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=2, min_samples_split=4)
tree2.fit(X_train, y_train)

y_pred_train2 = tree2.predict(X_train)
y_pred_test2 = tree2.predict(X_test)

train_accuracy2 = accuracy_score(y_train, y_pred_train2)
test_accuracy2 = accuracy_score(y_test, y_pred_test2)
train_precision2 = precision_score(y_train, y_pred_train2)
test_precision2 = precision_score(y_test, y_pred_test2)
train_recall2 = recall_score(y_train, y_pred_train2)
test_recall2 = recall_score(y_test, y_pred_test2)
train_f12 = f1_score(y_train, y_pred_train2)
test_f12 = f1_score(y_test, y_pred_test2)
train_roc_auc2 = roc_auc_score(y_train, y_pred_train2)
test_roc_auc2 = roc_auc_score(y_test, y_pred_test2)

print(f"Tree2 Train accuracy: {train_accuracy2:.4f}")
print(f"Tree2 Test accuracy: {test_accuracy2:.4f}")
print(f"Tree2 Train precision: {train_precision2:.4f}")
print(f"Tree2 Test precision: {test_precision2:.4f}")
print(f"Tree2 Train recall: {train_recall2:.4f}")
print(f"Tree2 Test recall: {test_recall2:.4f}")
print(f"Tree2 Train F1 score: {train_f12:.4f}")
print(f"Tree2 Test F1 score: {test_f12:.4f}")
print(f"Tree2 Train ROC AUC score: {train_roc_auc2:.4f}")
print(f"Tree2 Test ROC AUC score: {test_roc_auc2:.4f}")

# Save the best model and preprocessing tools
joblib.dump(best_model, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(encoders, 'encoders.joblib')

print("Model training and evaluation completed and saved.")

# Plot and save the decision tree of the best model
plt.figure(figsize=(20,10))
plot_tree(best_model, feature_names=X_train.columns, class_names=['Class 0', 'Class 1'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.savefig('best_model_tree.png')
plt.show()
