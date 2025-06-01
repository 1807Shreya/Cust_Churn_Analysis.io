import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df=pd.read_csv("Climate Change & Global Temperature Analysis.csv",low_memory=False)
'''print(df.head())
print(df.isnull().sum())
print(df.info())
print(df.describe())'''



df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#print(df['Date'].dtype)
#print(df[['Date']].head())

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
#print(df[['Date', 'Year']].head(15))
df.to_csv("climate_data_updated.csv", index=False)

# Average global temperature by year
temp_by_year = df.groupby('Year')['AverageTemperature'].mean().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=temp_by_year, x='Year', y='AverageTemperature')
plt.title("Average Global Temperature Over Years")
plt.xlabel("Year")
plt.ylabel("Average Temperature (°C)")
plt.grid(True)
#plt.show()

# Total global emissions by year
emissions_by_year = df.groupby('Year')['Total'].sum().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=emissions_by_year, x='Year', y='Total')
plt.title("Total Global CO2 Emissions Over Years")
plt.xlabel("Year")
plt.ylabel("Total CO2 Emissions")
plt.grid(True)
#plt.show()

# Group by year for avg temp and total emissions
yearly_data = df.groupby('Year').agg({'AverageTemperature':'mean', 'Total':'sum'}).reset_index()

plt.figure(figsize=(8,6))
sns.scatterplot(data=yearly_data, x='Total', y='AverageTemperature')
plt.title("Correlation: Total CO2 Emissions vs Average Temperature")
plt.xlabel("Total CO2 Emissions")
plt.ylabel("Average Temperature (°C)")
#plt.show()

# Calculate correlation coefficient
correlation = yearly_data['Total'].corr(yearly_data['AverageTemperature'])
#print(f"Correlation coefficient: {correlation:.3f}")




# Aggregate by Year: mean temperature and sum of total emissions
yearly_data = df.groupby('Year').agg({
    'AverageTemperature': 'mean',
    'Total': 'sum'  # total CO2 emissions
}).reset_index()

# Drop rows with missing values
yearly_data = yearly_data.dropna()
print(yearly_data.head())

X = yearly_data[['Total']].values

# Target (y): Average Temperature
y = yearly_data['AverageTemperature'].values

# Initialize and train model
model = LinearRegression()
model.fit(X, y)

# Check model parameters
print(f"Coefficient (slope): {model.coef_[0]:.6f}")
print(f"Intercept: {model.intercept_:.6f}")

# Predict temperature using the model
y_pred = model.predict(X)

plt.figure(figsize=(10,6))
plt.scatter(X, y, label='Actual Temperature')
plt.plot(X, y_pred, color='red', label='Predicted Temperature')
plt.xlabel('Total CO2 Emissions')
plt.ylabel('Average Temperature (°C)')
plt.title('Linear Regression: CO2 Emissions vs Temperature')
plt.legend()
#plt.show()

# Example: predict for emission = 40 billion (adjust unit based on your data)
future_emission = np.array([[4e10]])  # Replace with realistic value

future_temp = model.predict(future_emission)
print(f"Predicted temperature for CO2 emissions of {future_emission[0][0]:.2e} is {future_temp[0]:.2f} °C")

# Save cleaned and processed data to a new CSV file
df.to_csv("climate_data_cleaned.csv", index=False)

print(" Cleaned data saved to 'climate_data_cleaned.csv'")