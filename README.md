# Importing the Pandas library
import pandas as pd

# Reading a CSV file into a DataFrame
dataset = pd.read_csv("//content/Salary.csv")
dataset.head()

# Assign features and target variable
x = dataset['YearsExperience'].values.reshape(-1, 1)  # Reshape for a single feature
y = dataset['Salary'].values
# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
 #Create an instance of StandardScaler
scaler = StandardScaler()
# Fit the scaler to the training data and transform it
x_train_scaled = scaler.fit_transform(x_train)
# Transform the test data using the same scaler
x_test_scaled = scaler.transform(x_test)

reg = LinearRegression()
reg.fit(x_train_scaled, y_train)

# Make predictions using the test data
y_pred = reg.predict(x_test_scaled)

import pickle
from sklearn.metrics import r2_score # Import r2_score

# Save the trained model
with open('salary_reg.sav', 'wb') as f:
    pickle.dump(reg, f)

# Save the fitted scaler
with open('scaler.sav', 'wb') as f:
    pickle.dump(scaler, f)


# Calculate the R² score
r2 = r2_score(y_test, y_pred)
print(f"R-squared value = {r2:.4f}")
