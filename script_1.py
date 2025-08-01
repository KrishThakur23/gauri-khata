# Create requirements.txt
requirements_content = """streamlit==1.28.1
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
matplotlib==3.7.2
plotly==5.17.0
seaborn==0.12.2
joblib==1.3.2
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

print("Created requirements.txt")
print("Contents:")
print(requirements_content)