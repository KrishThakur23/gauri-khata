# Create model_training.py
model_training_content = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the Telco Customer Churn dataset"""
    print("Loading Telco Customer Churn dataset...")
    df = pd.read_csv('Telco-Customer-Churn.csv')
    
    print(f"Dataset loaded: {df.shape[0]} customers, {df.shape[1]} features")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    
    # Drop customer ID
    df = df.drop('customerID', axis=1)
    
    # Handle TotalCharges - convert to numeric, replace spaces with NaN
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Feature Engineering
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                               labels=['0-12', '13-24', '25-48', '49-72'])
    
    df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['charges_per_service'] = df['MonthlyCharges'] / (
        (df['PhoneService'] == 'Yes').astype(int) +
        (df['InternetService'] != 'No').astype(int) +
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport'] == 'Yes').astype(int) +
        (df['StreamingTV'] == 'Yes').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int) + 1
    )
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    
    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    return df

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
    
    print(f"\\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def train_models():
    """Train multiple models and save the best one"""
    print("Starting model training pipeline...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    print(f"\\nFeatures shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    }
    
    results = []
    trained_models = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\\n{'='*50}")
        print(f"Training {name}...")
        
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            result = evaluate_model(model, X_test_scaled, y_test, name)
            trained_models[name] = {'model': model, 'scaled': True}
        else:
            model.fit(X_train, y_train)
            result = evaluate_model(model, X_test, y_test, name)
            trained_models[name] = {'model': model, 'scaled': False}
        
        results.append(result)
    
    # Find best model based on AUC
    best_model_result = max(results, key=lambda x: x['auc'])
    best_model_name = best_model_result['model_name']
    
    print(f"\\n{'='*50}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Best AUC: {best_model_result['auc']:.4f}")
    print(f"Best Accuracy: {best_model_result['accuracy']:.4f}")
    
    # Save models and scaler
    for name, model_info in trained_models.items():
        filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model_info['model'], filename)
        print(f"Saved {filename}")
    
    joblib.dump(scaler, 'scaler.pkl')
    print("Saved scaler.pkl")
    
    # Save feature names
    joblib.dump(list(X.columns), 'feature_names.pkl')
    print("Saved feature_names.pkl")
    
    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'XGBoost']:
        best_model = trained_models[best_model_name]['model']
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\\nTop 10 Most Important Features ({best_model_name}):")
        print(feature_importance.head(10))
        
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("Saved feature_importance.csv")
    
    return results, trained_models, best_model_name

if __name__ == "__main__":
    try:
        results, models, best_model = train_models()
        print(f"\\n{'='*60}")
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best performing model: {best_model}")
        print("All models and preprocessing objects saved.")
        print("Ready to run: streamlit run app.py")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
'''

with open('model_training.py', 'w') as f:
    f.write(model_training_content)

print("Created model_training.py")
print("File size:", len(model_training_content), "characters")