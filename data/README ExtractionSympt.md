This code performs symptom extraction from medical text using SpaCy and SciSpaCy for processing medical language in our PMC dataset. Here's a breakdown of what each part does:

### 1. **Libraries Import**:
   - `spacy`: Main library used for Natural Language Processing (NLP).
   - `pandas`: Used for handling tabular data (e.g., CSV files).
   - `scispacy`: Specialized version of SpaCy optimized for medical text.

### 2. **Model Loading**:
   - `nlp = spacy.load("en_core_sci_sm")`: This loads a pre-trained model (`en_core_sci_sm`), which is a small version of SciSpaCy, optimized for medical texts. This model is used to process and analyze medical text.

### 3. **Example Text**:
   - `text = "The patient shows signs of fever, continuous coughing, and shortness of breath."`: This is an example input text containing symptoms like fever, cough, and shortness of breath.

### 4. **Symptom List**:
   - `symptom_list = [...]`: A list containing common medical symptoms (e.g., "fever", "cough", "shortness of breath", etc.). These are the symptoms the code will look for in the provided text.

### 5. **Symptom Extraction Function**:
   - `extract_symptoms`: This function processes the input `text` with the SciSpaCy model to detect entities (i.e., medical terms or symptoms) and checks if they match any of the symptoms from the `symptom_list`. If a match is found, the symptom is added to the `extracted_symptoms` list.

### 6. **Loading the Dataset**:
   - `df = pd.read_csv('/content/PMC-Patients-oa-9995.csv')`: This loads a CSV file containing medical records, where each row may contain some medical text (e.g., symptoms or diagnoses). The column with the medical text is named `patient`.

### 7. **Applying the Symptom Extraction**:
   - `df['extracted_symptoms'] = df['patient'].apply(lambda x: extract_symptoms(x, symptom_list))`: This line applies the `extract_symptoms` function to each row of the `patient` column, extracting symptoms based on the predefined `symptom_list`. The results are stored in a new column `extracted_symptoms`.

### 8. **Saving the Results**:
   - `df.to_csv('extracted_symptoms.csv', index=False)`: This saves the updated dataset with the extracted symptoms to a new CSV file named `extracted_symptoms.csv`.

### 9. **Displaying the Results**:
   - `print(df[['patient_uid', 'extracted_symptoms']])`: This prints the `patient_uid` (a unique identifier for each patient) and the corresponding `extracted_symptoms` for each row.
