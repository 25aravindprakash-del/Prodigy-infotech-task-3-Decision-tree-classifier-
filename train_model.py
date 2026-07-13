import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_fscore_support

def train_and_export():
    print("Loading data...")
    # Load dataset
    data_path = os.path.join("bank-additional", "bank-additional-full.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("bank", "bank-full.csv") # Fallback
    
    # Semicolon separator
    df = pd.read_csv(data_path, sep=";")
    
    print(f"Data shape: {df.shape}")
    
    # 1. Data Cleaning
    # Discard duration as it leaks the target (as recommended in dataset description)
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
        print("Dropped 'duration' column to prevent target leakage.")
    
    # Target mapping
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Extract categories and their options for categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    feature_categories = {}
    for col in categorical_cols:
        feature_categories[col] = sorted(df[col].unique().tolist())
        
    print("Categorical features and unique values:")
    for k, v in feature_categories.items():
        print(f"  {k}: {v}")
        
    # Split into features and target
    X = df.drop(columns=['y'])
    y = df['y']
    
    # Store numerical column names
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    print(f"Numerical features: {numerical_cols}")
    
    # One-hot encode X
    X_encoded = pd.get_dummies(X, columns=categorical_cols)
    encoded_feature_names = X_encoded.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Tuning Decision Tree Classifier...")
    # Setup grid search for decision tree parameters
    param_grid = {
        'max_depth': [3, 4, 5, 6, 8, 10],
        'min_samples_leaf': [10, 20, 50, 100],
        'criterion': ['entropy', 'gini']
    }
    
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1', # optimize for F1 score since target is imbalanced
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    print(f"Best parameters: {grid.best_params_}")
    
    # Evaluate
    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    conf_mat = confusion_matrix(y_test, y_pred).tolist()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Feature importance
    importances = best_clf.feature_importances_
    feat_imp = sorted(
        [{"feature": name, "importance": float(imp)} for name, imp in zip(encoded_feature_names, importances) if imp > 0.001],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    # Export Tree to Dict
    def tree_to_dict(tree, node_id=0):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        val = tree.value[node_id][0].tolist()
        
        node_dict = {
            "node_id": int(node_id),
            "samples": int(tree.n_node_samples[node_id]),
            "value": val
        }
        
        if left_child == right_child:  # Leaf
            class_idx = int(np.argmax(val))
            node_dict["is_leaf"] = True
            node_dict["class_idx"] = class_idx
            node_dict["class_name"] = "yes" if class_idx == 1 else "no"
        else:  # Internal split
            node_dict["is_leaf"] = False
            node_dict["feature_idx"] = int(tree.feature[node_id])
            node_dict["feature_name"] = str(encoded_feature_names[tree.feature[node_id]])
            node_dict["threshold"] = float(tree.threshold[node_id])
            node_dict["left"] = tree_to_dict(tree, left_child)
            node_dict["right"] = tree_to_dict(tree, right_child)
            
        return node_dict

    tree_dict = tree_to_dict(best_clf.tree_)
    
    # Sample subset of ROC curve points for smaller size
    roc_points = []
    step = max(1, len(fpr) // 100)
    for i in range(0, len(fpr), step):
        roc_points.append({"fpr": float(fpr[i]), "tpr": float(tpr[i])})
    # Make sure end points are included
    if roc_points[-1]["fpr"] != 1.0 or roc_points[-1]["tpr"] != 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})
        
    output_data = {
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        },
        "confusion_matrix": conf_mat,
        "feature_importances": feat_imp,
        "feature_categories": feature_categories,
        "numerical_features": numerical_cols,
        "encoded_features": encoded_feature_names,
        "roc_curve": roc_points,
        "decision_tree": tree_dict
    }
    
    # Write to file
    with open("model_data.json", "w") as f:
        json.dump(output_data, f, indent=2)
        
    print("Model assets exported successfully to model_data.json")

if __name__ == "__main__":
    train_and_export()
