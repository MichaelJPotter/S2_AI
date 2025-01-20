
#pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas data frames) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
  
# metadata 
print(cdc_diabetes_health_indicators.metadata) 
  
# variable information 
print(cdc_diabetes_health_indicators.variables) 

# Check the shape of the data
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Display the first few rows of features and targets
print(X.head())  # Features
print(y.head())  # Target

# Check for missing values
print(X.isnull().sum())  # Missing values in features
print(y.isnull().sum())  # Missing values in target

# Summary statistics of numerical features
print(X.describe())
