import pandas as pd
import ast  # To safely evaluate the string representation of a dictionary
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
data = pd.read_csv('/Users/gaurika/Documents/Masters/Classes/Semester 1/BioStat and AI in med/Project/medLLM/data/pmc_patients/PMC-Patients-oa-9995.csv')

# Display the first few rows to verify the content
print(data.head())

# Convert the 'relevant_articles' column to a dictionary if it's stored as a string
data['relevant_articles'] = data['relevant_articles'].apply(ast.literal_eval)

# Check the first element to verify conversion
print(data['relevant_articles'].iloc[0])

# Define a function to count the number of relevant articles
def count_relevant_articles(article_dict):
    return len(article_dict)

# Apply the function to the dataframe
data['num_relevant_articles'] = data['relevant_articles'].apply(count_relevant_articles)

# Output the results
print(data[['patient_uid', 'num_relevant_articles']])

# Group by the number of articles and count patients in each group
article_patient_counts = data.groupby('num_relevant_articles').size().reset_index(name='num_patients')

# Plot the trend
plt.figure(figsize=(10, 6))
plt.plot(article_patient_counts['num_relevant_articles'], article_patient_counts['num_patients'], marker='o', linestyle='-')
plt.title('Number of Articles vs. Number of Patients', fontsize=16)
plt.xlabel('Number of Relevant Articles', fontsize=14)
plt.ylabel('Number of Patients', fontsize=14)
plt.grid(True)
plt.show() 

# Basic Analysis 
print(data.columns.tolist())

# Calculate the length of patient summaries
data['summary_length'] = data['patient'].apply(lambda x: len(str(x)))

# Plot distribution of summary lengths
sns.histplot(data['summary_length'], bins=30, kde=True, color='green')
plt.title("Patient Summary Length Distribution")
plt.xlabel("Length of Summary- characters")
plt.ylabel("Number of Patients")
plt.show()

# Count and visualize gender
gender_counts = data['gender'].value_counts()
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

#Summary Length vs Relevant Articles
sns.scatterplot(data=data, x='summary_length', y='num_relevant_articles', color='brown')
plt.title("Summary Length vs Number of Relevant Articles")
plt.xlabel("Summary Length")
plt.ylabel("Number of Relevant Articles")
plt.show()

# Extract the number of similar patients
data['num_similar_patients'] = data['similar_patients'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)

# Plot distribution of similar patients
sns.histplot(data['num_similar_patients'], bins=20, kde=False, color='orange')
plt.title("Distribution of Similar Patients")
plt.xlabel("Number of Similar Patients")
plt.ylabel("Frequency")
plt.show()

# Correlation between relevant articles and similar patients
sns.scatterplot(data=data, x='num_relevant_articles', y='num_similar_patients', color='teal')
plt.title("Relevant Articles vs Similar Patients")
plt.xlabel("Number of Relevant Articles")
plt.ylabel("Number of Similar Patients")
plt.show()

def extract_age(age_data):
    try:
        # If the data is in the format [[num, 'unit']], extract the numeric value
        if isinstance(age_data, list) and isinstance(age_data[0], list) and isinstance(age_data[0][0], (int, float)):
            return age_data[0][0]
    except:
        pass
    return np.nan

# Apply the function to the 'age' column to create 'age_cleaned'
print(data['age'].head(10))
import re

def extract_age_v2(age_data):
    try:
        # If the data is a list, check if it contains a sublist with numeric value
        if isinstance(age_data, list):
            for item in age_data:
                if isinstance(item, list) and len(item) > 0:
                    # Extract the numeric value (using regex to handle different formats)
                    number = re.findall(r"[-+]?\d*\.\d+|\d+", str(item[0]))
                    if number:
                        return float(number[0])
        # If the data is already a string or other types, try to extract any numeric value
        elif isinstance(age_data, str):
            number = re.findall(r"[-+]?\d*\.\d+|\d+", age_data)
            if number:
                return float(number[0])
    except:
        pass
    return np.nan

# Apply this robust function to the 'age' column
data['age_cleaned'] = data['age'].apply(extract_age_v2)

# Display the first few rows to verify the extraction
print(data[['age', 'age_cleaned']].head(10))

# Plot age distribution
sns.histplot(data['age_cleaned'], bins=20, kde=True, color='blue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Age vs number of relevant articles
sns.scatterplot(data=data, x='age_cleaned', y='num_relevant_articles', hue='gender', palette='coolwarm')
plt.title("Age vs Number of Relevant Articles")
plt.xlabel("Age")
plt.ylabel("Number of Relevant Articles")
plt.legend(title='Gender')
plt.show()