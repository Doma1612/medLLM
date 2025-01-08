import pandas as pd
import ast  # To safely evaluate the string representation of a dictionary
import matplotlib.pyplot as plt
import seaborn as sns


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