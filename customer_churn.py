import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as wr
wr.filterwarnings('ignore')
import random
'''from datetime import datetime, timedelta

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate 50,000 entries
num_entries = 50000

# Customers data
customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, num_entries + 1)]
genders = np.random.choice(["Male", "Female"], size=num_entries)
ages = np.random.randint(18, 70, size=num_entries)
regions = np.random.choice(["North", "South", "East", "West"], size=num_entries)
tenures = np.random.randint(1, 61, size=num_entries)
service_plans = np.random.choice(["Basic", "Standard", "Premium"], size=num_entries)

df_customers = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Gender": genders,
    "Age": ages,
    "Region": regions,
    "Tenure": tenures,
    "Service_Plan": service_plans
})

# Usage data
monthly_usage = np.round(np.random.normal(loc=30, scale=10, size=num_entries), 1)
avg_call_duration = np.round(np.random.normal(loc=4, scale=1.5, size=num_entries), 1)
num_support_calls = np.random.poisson(lam=2, size=num_entries)

df_usage = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Monthly_Usage_GB": monthly_usage,
    "Avg_Call_Duration": avg_call_duration,
    "Num_Support_Calls": num_support_calls
})

# Churn data
churn_flags = np.random.choice(["Yes", "No"], size=num_entries, p=[0.2, 0.8])
churn_dates = [
    (datetime.today() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d') if flag == "Yes" else ""
    for flag in churn_flags
]

df_churn = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Churn_Flag": churn_flags,
    "Churn_Date": churn_dates
})

# Save to Excel
with pd.ExcelWriter("Telecom_Churn_50000.xlsx", engine='xlsxwriter') as writer:
    df_customers.to_excel(writer, sheet_name='Customers', index=False)
    df_usage.to_excel(writer, sheet_name='Usage', index=False)
    df_churn.to_excel(writer, sheet_name='Churn', index=False)

print("Telecom_Churn_50000.xlsx has been created successfully.")'''

#1.	Data Cleaning & Preparation
customers = pd.read_excel("Telecom_Churn_50000.xlsx", sheet_name="Customers")
usage = pd.read_excel("Telecom_Churn_50000.xlsx", sheet_name="Usage")
churn = pd.read_excel("Telecom_Churn_50000.xlsx", sheet_name="Churn")

merged_df=pd.merge(customers, usage, on="Customer_ID", how='left')
merged_df=pd.merge(merged_df, churn, on="Customer_ID",how='left')

#print(merged_df.isnull().sum())

missing_date_mismatch=merged_df[(merged_df["Churn_Date"].isna())&(merged_df["Churn_Flag"]=="Yes")]
#print(f"Churned customers with missing dates:{len(missing_date_mismatch)}")

merged_df["Churn_Date_Fill"]=merged_df["Churn_Date"].fillna("Not Applicable")

merged_df.to_excel("Cleaned_telecom_data_50000.xlsx",index=False)

#2.	Feature Engineering
#Create flags for high usage, long tenure, frequent support calls.

df=pd.read_excel("Cleaned_telecom_data_50000.xlsx")
#print(df.head())

df["High_Usage_Flag"]=(df["Monthly_Usage_GB"]>50).astype(int)
df["Long_Tenure_Flag"]=(df["Tenure"]>24).astype(int)
df["Frequent_Support_Flag"]=(df["Num_Support_Calls"]>3).astype(int)

#print(df[["Monthly_Usage_GB", "Tenure", "Num_Support_Calls",
          #"High_Usage_Flag", "Long_Tenure_Flag", "Frequent_Support_Flag"]].head())

df.to_excel("Cleaned_telecom_data_50000.xlsx",index=False)

#3.	Exploratory Analysis
df=pd.read_excel("Cleaned_telecom_data_50000.xlsx")
df["Churn_Flag"]=df["Churn_Flag"].map({"Yes":1,"No":0})

#Tenure Bins
df["Tenure_Group"]=pd.cut(df["Tenure"],bins=[0,12,24,36,60],labels=["0-12","13-24","25-36","37-60"])

tenure_churn=df.groupby("Tenure_Group")["Churn_Flag"].mean().reset_index()

sns.barplot(x='Tenure_Group',y='Churn_Flag',data=tenure_churn)
plt.title("churn Rate by Tenure Group")
plt.ylabel("Churn Rate")
plt.xlabel("Tenure(Months)")
#plt.show()

#Service Plan
plan_churn=df.groupby("Service_Plan")["Churn_Flag"].mean().reset_index()
sns.barplot(x="Service_Plan",y="Churn_Flag",data=plan_churn)
plt.title("Churn Rate by Service Plan")
plt.ylabel("Churn Rate")
plt.xlabel("Service Plan")
plt.xticks(rotation=45)
#plt.show()

#Age Group
df["Age_Group"]=pd.cut(df["Age"],bins=[18,30,45,60,100], labels=["18-30","31-45","46-60","60+"])
age_churn=df.groupby("Age_Group")["Churn_Flag"].mean().reset_index()

sns.barplot(x="Age_Group",y="Churn_Flag",data=age_churn)
plt.title("Churn Rate by Age Group")
plt.ylabel("Churn Rate")
plt.xlabel("Age Group")
#plt.show()

# Correlation Between Churn and Usage Behavior (Select numeric features to check correlation with churn)
numeric_features = ["Monthly_Usage_GB", "Avg_Call_Duration", "Num_Support_Calls", 
                    "High_Usage_Flag", "Frequent_Support_Flag", "Long_Tenure_Flag", "Churn_Flag"]

corr = df[numeric_features].corr()

# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
#plt.show()

#df['Churn_Flag'] = df['Churn_Flag'].replace({0: 'Retained', 1: 'Churned'})
df["Churn_Flag"]=df["Churn_Flag"].map({1:"Yes",0:"No"})
#print(df['Churn_Flag'].value_counts())

df.to_excel("Cleaned_telecom_data_50000.xlsx",index=False)

