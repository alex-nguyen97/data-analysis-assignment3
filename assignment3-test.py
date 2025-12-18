# read the file from train_subset(in).csv and write the output to train_subset(out).csv
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score


def remove_id_column(df):
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
    return df


def preprocess_data(df):
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].mean())

    # Binarize the satisfaction column
    if 'satisfaction' in df.columns:
        df['satisfaction'] = df['satisfaction'].map({
            'satisfied': 1,
            'neutral or dissatisfied': 0
        })

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def get_numerical_summary(df):
    # Select only numeric columns
    num_cols = df.select_dtypes(include=[np.number])

    # Basic summary (with percentiles)›
    summary = num_cols.describe(percentiles=[.25, .5, .75, .9])

    # Add variance row (rounded to 2 decimals)
    summary.loc['variance'] = num_cols.var()
    summary.loc['median'] = num_cols.median()

    # get values counts for each column
    for col in num_cols.columns:
        mode = num_cols[col].mode()[0]
        mode_count = (num_cols[col] == mode).sum()
        summary.at['mode', col] = mode
        summary.at['mode count', col] = mode_count
        summary.at['skewness', col] = stats.skew(num_cols[col])
        summary.at['kurtosis', col] = stats.kurtosis(num_cols[col])
        summary.at['range', col] = num_cols[col].max() - num_cols[col].min()
        summary.at['missing count', col] = num_cols[col].isnull().sum()
        summary.at['missing %', col] = round(
            (num_cols[col].isnull().mean() * 100), 2)
        summary.at['unique count', col] = num_cols[col].nunique()

    # round all values to 2 decimals
    summary = summary.round(2)
    return summary


def plot_numerical_features(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols].hist(bins=10, figsize=(20, 15))
    plt.savefig("32130_AT2_25608100_histograms.png")
    plt.show()


def get_categorical_summary(df):
    cat_cols = df.select_dtypes(include=['object', 'category'])
    summary_list = []
    # get the list of unique values and their counts for each column put into summary list
    for col in cat_cols.columns:
        value_counts = cat_cols[col].value_counts()
        for value, count in value_counts.items():
            missing_count = cat_cols[col].isnull().sum()
            missing_percent = round(
                (cat_cols[col].isnull().mean() * 100), 2)
            summary_list.append({
                'Feature': col,
                'Value': value,
                'Count': count,
                'Missing Count': missing_count,
                'Missing %': missing_percent
            })

    summary_df = pd.DataFrame(summary_list).set_index('Feature')
    return summary_df


def plot_categorical_features(df):
    cat_cols = df.select_dtypes(include=['object', 'category'])
    # plot all pie charts at once
    n_cols = 3  # number of charts per row
    n_rows = (len(cat_cols.columns) + n_cols -
              1) // n_cols  # auto compute rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()  # make it 1D for easy iteration

    for i, col in enumerate(cat_cols.columns):
        ax = axes[i]
        value_counts = cat_cols[col].value_counts()
        value_counts.plot.pie(
            ax=ax, autopct='%1.1f%%', startangle=90, ylabel='', legend=False
        )
        ax.set_title(f'Value Distribution for {col}', fontsize=12)

    # remove unused subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_missing_rows(df):
    # Combine into a single table
    # Check for missing values per column
    missing_counts = df.isnull().sum()

    # Also check percentage of missing values
    missing_percent = (df.isnull().mean() * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing Values': missing_counts,
        'Missing %': missing_percent
    })

    # draw a bar chart for missing values
    missing_summary = missing_summary['Missing Values']
    if not missing_summary.empty:
        missing_summary.plot.bar(
            title='Missing Values per Feature', figsize=(10, 6))
        plt.ylabel('Count of Missing Values')
        plt.show()


def plot_outlier_info(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    outlier_columns = []

    # Detect outliers and print only when found
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        if not outliers.empty:
            print(f"{col}: {len(outliers)} outliers detected")
            outlier_columns.append(col)

    # Plot all boxplots together (only for columns with outliers)
    if outlier_columns:
        plt.figure(figsize=(5 * len(outlier_columns), 5))
        df[outlier_columns].boxplot(
            flierprops=dict(marker='o', color='red', markersize=5)
        )
        plt.title("Box Plots for Columns with Outliers")
        plt.show()
    else:
        print("No outliers detected in any numeric column.")


# Read the input CSV file
df = pd.read_csv('train_subset(in).csv')
# Display the first few rows of the dataframe
remove_id_column(df)

# plot_numerical_features(df)
# plot_categorical_features(df)
# plot_missing_rows(df)
# plot_outlier_info(df)

# # print duplicate rows
# duplicate_rows = df[df.duplicated()]
# print(f"Number of duplicate rows: {len(duplicate_rows)}")
preprocess_data(df)


# # show the heatmap of correlation matrix
# plt.figure(figsize=(12, 10))
# correlation_matrix = X.corr()
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title("Feature Correlation Matrix")
# plt.show()

df.drop("Arrival Delay in Minutes", axis=1, inplace=True)

# # normalize all features for better SVM convergence
X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(
    X), columns=X.columns, index=X.index)
X = X_scaled

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)

print(X.head())

# ============================
# 3. Train classifiers
# ============================


# Define hyperparameter grids with explanations
param_grids = {
    "Logistic Regression": {
        # Type of regularization (L1 removes useless features, L2 shrinks weights)
        "penalty": ["l1", "l2"],
        # How strongly to regularize (smaller = stronger regularization)
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "saga"],  # Algorithm used to find best weights
        "max_iter": [500, 1000]  # How many steps to try before stopping
    },
    "Decision Tree": {
        # Maximum levels of the tree (None = unlimited)
        "max_depth": [3, 5, 7, 10],
        # Minimum samples required to split a node
        "min_samples_split": [2, 5, 10],
        # Minimum samples required at a leaf (end point)
        "min_samples_leaf": [1, 2, 4],
        # How the tree decides best split (gini or info gain)
        "criterion": ["gini", "entropy"]
    },
    "Random Forest": {
        "n_estimators": [100, 200],  # Number of trees in the forest
        "max_depth": [3, 5, 7],  # Max levels in each tree
        "min_samples_split": [2, 5],  # Minimum samples to split a node
        "min_samples_leaf": [2, 4],  # Minimum samples at a leaf node
    },
    "XGBoost": {
        "n_estimators": [100, 200],  # Number of boosting rounds (trees)
        "max_depth": [3, 5, 7],  # Maximum depth of each tree
        # How fast the model learns (smaller = slower but safer)
        "learning_rate": [0.1, 0.2],
        # Fraction of data used per tree (helps prevent overfitting)
        "subsample": [0.8, 1.0],
    },
    "K-Nearest Neighbors": {
        # Number of nearest neighbors to consider for voting
        "n_neighbors": [3, 5, 7, 10],
        # Equal votes or closer neighbors count more
        "weights": ["uniform", "distance"],
        "p": [1, 2]  # Distance metric: 1 = Manhattan, 2 = Euclidean
    },
    "Naive Bayes": {
        # Small number added to variance for stability
        "var_smoothing": [1e-9, 1e-8, 1e-7]
    },
    "Bagging Classifier": {
        "n_estimators": [50, 100],  # Number of base models to combine
        # Fraction of samples used per base model
        "max_samples": [0.5, 0.7],
        # Fraction of features used per base model
        "max_features": [0.5, 0.7]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],  # Number of boosting rounds
        "learning_rate": [0.01, 0.1, 1.0],  # Shrink each tree's contribution
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],  # Number of boosting rounds
        "learning_rate": [0.01, 0.1, 0.2],  # How fast the model learns
        "min_samples_split": [2, 5, 10]  # Minimum samples to split a node
    }
}

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Bagging Classifier": BaggingClassifier(estimator=DecisionTreeClassifier()),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier()),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Run GridSearchCV for all models
results = {}
best_models = {}

for name, model in models.items():
    # add timer here to measure training time
    start_time = time.time()
    print(f"\n▶ Running GridSearchCV for {name}...")
    try:
        # GridSearchCV to find best parameters
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids.get(name, {}),
            scoring="roc_auc",  # CV is still based on ROC AUC
            cv=5,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        # Best model from grid search
        best_model = grid_search.best_estimator_
        best_models[name] = best_model

        # Predictions
        y_pred = best_model.predict(X_test)

        # Probabilities for ROC-AUC (if available)
        try:
            y_prob = best_model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = None

        # Confusion matrix for specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Compute metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Error Rate": 1 - accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall (Sensitivity)": recall_score(y_test, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None
        }
        # Store results
        results[name] = {
            "Best Parameters": grid_search.best_params_,
            **metrics
        }
        print(f"  Best Parameters: {grid_search.best_params_}")
        print(f"✓ {name} evaluated. Test ROC-AUC: {metrics['ROC-AUC']}")

    except Exception as e:
        print(f"⚠️ {name} failed: {e}")
    end_time = time.time()
    print("Training time:", round(end_time - start_time, 2), "seconds")

# plo

plt.figure(figsize=(12, 9))
for i, (name, model) in enumerate(best_models.items()):
    try:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(3, 3, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    except Exception as e:
        print(f"⚠️ Could not plot confusion matrix for {name}: {e}")
plt.tight_layout()
plt.show()

# ==============================================
# 4. ROC Curve for All Models (with Best Params)
# ==============================================
plt.figure(figsize=(12, 9))
for name, model in best_models.items():
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        params = results[name]["Best Parameters"]
        label_text = f"{name} (AUC={roc_auc:.2f})\n{params}"
        plt.plot(fpr, tpr, label=label_text)
    except Exception as e:
        print(f"⚠️ Could not plot ROC for {name}: {e}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve with Best Hyperparameters', fontsize=14)
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.show()

# ============================
# 5. Show results
# ============================
results_df = pd.DataFrame(results).T
print(results_df)
df_test_raw = pd.read_csv('test_kaggle_features(in).csv')
preprocess_data(df_test_raw)
df_test = df_test_raw.copy()
remove_id_column(df_test)

df_test.drop("Arrival Delay in Minutes", axis=1, inplace=True)

X_test_kaggle_scaled = pd.DataFrame(
    scaler.transform(df_test),
    columns=df_test.columns,
    index=df_test.index
)

best_model_name = results_df['ROC-AUC'].idxmax()
best_model = best_models[best_model_name]
best_params = results[best_model_name]['Best Parameters']

print(f"\n🏆 Best Model Selected: {best_model_name}")
print(f"🔧 Best Hyperparameters: {best_params}")

y_test_kaggle_pred = best_model.predict(X_test_kaggle_scaled)
df_test_raw['satisfaction'] = y_test_kaggle_pred
df_test_raw['satisfaction'] = df_test_raw['satisfaction'].map({
    1: "satisfied",
    0: "neutral or dissatisfied"
})

# Save to CSV
df_test_raw.to_csv('test_features(out).csv', index=False)
print("✅ Predictions saved to 'test_features(out).csv'")

# # create a new df taking df_test_raw id and df_test_raw satisfaction
# df_test_satisfaction = df_test_raw[['id', 'satisfaction']].copy()
# df_test_satisfaction.to_csv('test_kaggle_features(out).csv', index=False)
