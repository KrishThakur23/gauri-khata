import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate synthetic Telco Customer Churn dataset
n_customers = 7043

# Generate customer IDs
customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n_customers + 1)]

# Demographics
genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.5, 0.5])
senior_citizens = np.random.choice([0, 1], n_customers, p=[0.84, 0.16])
partners = np.random.choice(['Yes', 'No'], n_customers, p=[0.52, 0.48])
dependents = np.random.choice(['Yes', 'No'], n_customers, p=[0.30, 0.70])

# Account information
tenure = np.random.exponential(scale=24, size=n_customers).astype(int)
tenure = np.clip(tenure, 1, 72)  # Cap at 72 months

contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                 n_customers, p=[0.55, 0.21, 0.24])

paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41])

payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                  n_customers, p=[0.34, 0.23, 0.22, 0.21])

# Services
phone_service = np.random.choice(['Yes', 'No'], n_customers, p=[0.91, 0.09])
multiple_lines = np.where(phone_service == 'Yes', 
                         np.random.choice(['Yes', 'No'], n_customers, p=[0.53, 0.47]),
                         'No phone service')

internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22])

# Internet-dependent services
online_security = np.where(internet_service == 'No', 'No internet service',
                          np.random.choice(['Yes', 'No'], n_customers, p=[0.29, 0.71]))

online_backup = np.where(internet_service == 'No', 'No internet service',
                        np.random.choice(['Yes', 'No'], n_customers, p=[0.34, 0.66]))

device_protection = np.where(internet_service == 'No', 'No internet service',
                           np.random.choice(['Yes', 'No'], n_customers, p=[0.34, 0.66]))

tech_support = np.where(internet_service == 'No', 'No internet service',
                       np.random.choice(['Yes', 'No'], n_customers, p=[0.29, 0.71]))

streaming_tv = np.where(internet_service == 'No', 'No internet service',
                       np.random.choice(['Yes', 'No'], n_customers, p=[0.38, 0.62]))

streaming_movies = np.where(internet_service == 'No', 'No internet service',
                           np.random.choice(['Yes', 'No'], n_customers, p=[0.38, 0.62]))

# Charges
monthly_charges = np.random.normal(65, 30, n_customers)
monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

# Total charges based on tenure and monthly charges with some noise
total_charges = monthly_charges * tenure + np.random.normal(0, 200, n_customers)
total_charges = np.maximum(total_charges, monthly_charges)  # Ensure total >= monthly

# Some customers have missing total charges (like new customers)
missing_indices = np.random.choice(n_customers, size=11, replace=False)
total_charges_str = total_charges.astype(str)
total_charges_str[missing_indices] = ' '

# Generate churn based on realistic patterns
churn_prob = 0.1  # Base probability

# Increase churn probability based on factors
churn_prob_individual = np.full(n_customers, churn_prob)

# Month-to-month contracts have higher churn
churn_prob_individual = np.where(contract_types == 'Month-to-month', 
                                churn_prob_individual * 2.5, churn_prob_individual)

# High monthly charges increase churn
churn_prob_individual = np.where(monthly_charges > 80, 
                                churn_prob_individual * 1.5, churn_prob_individual)

# Low tenure increases churn
churn_prob_individual = np.where(tenure < 6, 
                                churn_prob_individual * 2.0, churn_prob_individual)

# Fiber optic customers have higher churn (service issues)
churn_prob_individual = np.where(internet_service == 'Fiber optic', 
                                churn_prob_individual * 1.3, churn_prob_individual)

# Electronic check payment increases churn
churn_prob_individual = np.where(payment_methods == 'Electronic check', 
                                churn_prob_individual * 1.2, churn_prob_individual)

# Paperless billing increases churn slightly
churn_prob_individual = np.where(paperless_billing == 'Yes', 
                                churn_prob_individual * 1.1, churn_prob_individual)

# Cap probability at reasonable level
churn_prob_individual = np.clip(churn_prob_individual, 0, 0.8)

# Generate actual churn
churn = np.random.binomial(1, churn_prob_individual, n_customers)
churn = np.where(churn == 1, 'Yes', 'No')

# Create DataFrame
df = pd.DataFrame({
    'customerID': customer_ids,
    'gender': genders,
    'SeniorCitizen': senior_citizens,
    'Partner': partners,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract_types,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_methods,
    'MonthlyCharges': np.round(monthly_charges, 2),
    'TotalCharges': total_charges_str,
    'Churn': churn
})

# Save to CSV
df.to_csv('Telco-Customer-Churn.csv', index=False)

print(f"Generated Telco Customer Churn dataset with {len(df)} customers")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
print("\nDataset preview:")
print(df.head())
print("\nDataset info:")
print(df.info())