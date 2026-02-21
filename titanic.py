import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc

print("--- 1. Data Loading and Splitting ---")
# Load ONLY train.csv
df = pd.read_csv('train.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Split 'train.csv' into 80% training and 20% testing sets.")

print("\n--- 2. Data Cleaning ---")
# a & c. Handle missing values 
# Drop Cabin (Too many missing values)
X_train = X_train.drop(columns=['Cabin'])
X_test = X_test.drop(columns=['Cabin'])
print("Dropped 'Cabin' column due to high percentage of missing values.")

# Impute Age: Learn median from TRAIN only, apply to both train and test
age_median = X_train['Age'].median()
X_train['Age'] = X_train['Age'].fillna(age_median)
X_test['Age'] = X_test['Age'].fillna(age_median)
print(f"Imputed missing 'Age' values with train median ({age_median}).")

# Impute Embarked: Learn mode from TRAIN only, apply to both train and test
embarked_mode = X_train['Embarked'].mode()[0]
X_train['Embarked'] = X_train['Embarked'].fillna(embarked_mode)
X_test['Embarked'] = X_test['Embarked'].fillna(embarked_mode)
print(f"Imputed missing 'Embarked' values with train mode ('{embarked_mode}').")

# b. Address noisy/inconsistent values
cols_to_drop = ['Name', 'Ticket', 'PassengerId']
X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)
print("Dropped 'Name', 'Ticket', and 'PassengerId' to reduce noise.")

print("\n--- 3. Feature Engineering ---")
# c. Construct new features
X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1
X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch'] + 1

X_train['IsAlone'] = (X_train['FamilySize'] == 1).astype(int)
X_test['IsAlone'] = (X_test['FamilySize'] == 1).astype(int)
print("Created new features: 'FamilySize' and 'IsAlone'.")

# a. Apply transformations (log-scaling)
X_train['Log_Fare'] = np.log1p(X_train['Fare'])
X_test['Log_Fare'] = np.log1p(X_test['Fare'])
X_train = X_train.drop(columns=['Fare'])
X_test = X_test.drop(columns=['Fare'])
print("Applied log1p-scaling to 'Fare' column to reduce skewness.")

# b. Encode categorical variables (One-hot encoding)
# Temporarily combine to ensure dummy columns match perfectly between train and test splits
X_train['is_train'] = 1
X_test['is_train'] = 0
combined = pd.concat([X_train, X_test])
combined = pd.get_dummies(combined, columns=['Sex', 'Embarked'], drop_first=True)

X_train = combined[combined['is_train'] == 1].drop(columns=['is_train'])
X_test = combined[combined['is_train'] == 0].drop(columns=['is_train'])
print("Applied One-Hot Encoding to 'Sex' and 'Embarked'.")

# a. Apply transformations (Standardization)
scaler = StandardScaler()
# Fit strictly on X_train to prevent test data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Standardized numerical features using StandardScaler.")

print("\n--- 4. Model Training and Evaluation ---")
results = {}
fpr_tpr_auc = {}

def evaluate_model(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'CM': cm}
    fpr_tpr_auc[name] = (fpr, tpr, roc_auc)
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    return acc

# a. Naive Bayes with Laplace Smoothing
nb_alpha_1 = BernoulliNB(alpha=1.0)
nb_alpha_1.fit(X_train_scaled, y_train)
prob_nb_1 = nb_alpha_1.predict_proba(X_test_scaled)[:, 1]
pred_nb_1 = nb_alpha_1.predict(X_test_scaled)
evaluate_model("BernoulliNB (alpha=1.0)", y_test, pred_nb_1, prob_nb_1)

nb_alpha_01 = BernoulliNB(alpha=0.01)
nb_alpha_01.fit(X_train_scaled, y_train)
prob_nb_01 = nb_alpha_01.predict_proba(X_test_scaled)[:, 1]
pred_nb_01 = nb_alpha_01.predict(X_test_scaled)
evaluate_model("BernoulliNB (alpha=0.01)", y_test, pred_nb_01, prob_nb_01)

# b. Linear Regression (Threshold = 0.5)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
prob_lr = lr.predict(X_test_scaled)
pred_lr = (prob_lr >= 0.5).astype(int)
evaluate_model("Linear Regression", y_test, pred_lr, prob_lr)

# b.ii.1 Ridge Regression (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
prob_ridge = ridge.predict(X_test_scaled)
pred_ridge = (prob_ridge >= 0.5).astype(int)
evaluate_model("Ridge Regression (L2)", y_test, pred_ridge, prob_ridge)

# b.ii.2 LASSO Regression (L1)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
prob_lasso = lasso.predict(X_test_scaled)
pred_lasso = (prob_lasso >= 0.5).astype(int)
evaluate_model("LASSO Regression (L1)", y_test, pred_lasso, prob_lasso)

print("\n--- Visualizations ---")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

models_to_plot = [
    "BernoulliNB (alpha=1.0)", "BernoulliNB (alpha=0.01)", 
    "Linear Regression", "Ridge Regression (L2)", "LASSO Regression (L1)"
]

# Plot Confusion Matrices
for i, name in enumerate(models_to_plot):
    sns.heatmap(results[name]['CM'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f"{name}\nConfusion Matrix")
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("True Label")

axes[-1].axis('off')
plt.tight_layout()
plt.savefig('confusion_matrices_split.png')

# Plot ROC Curves
plt.figure(figsize=(10, 8))
for name in models_to_plot:
    fpr, tpr, roc_auc = fpr_tpr_auc[name]
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves on Split Test Data')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('roc_curves_split.png')
print("Saved visualization plots as 'confusion_matrices_split.png' and 'roc_curves_split.png'.")