import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('UFC Fight Statistics (July 2016 - Nov 2024).csv')

print(data.info())
print(data.head())

sns.set_style("whitegrid")

# Graph 1: Distribution of Fight Methods (How each fight ended)
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Fight Method', palette='viridis', 
              order=data['Fight Method'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Distribution of Fight Methods (How each fight ended)")
plt.xlabel("Fight Method")
plt.ylabel("Number of Fights")
plt.show()

# Graph 2: Number of Fights per Bout Category
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Bout', palette='coolwarm', 
              order=data['Bout'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Number of Fights per Bout Category")
plt.xlabel("Bout Category")
plt.ylabel("Number of Fights")
plt.show()

# Graph 5: Distribution of Rounds in Fights
plt.figure(figsize=(8, 6))
unique_rounds = sorted(data['Rounds'].dropna().unique())
sns.countplot(data=data, x='Rounds', palette='Set2', order=unique_rounds)
plt.title("Distribution of Rounds in Fights")
plt.xlabel("Number of Rounds")
plt.ylabel("Number of Fights")
plt.show()

print("Data visualization completed.")

