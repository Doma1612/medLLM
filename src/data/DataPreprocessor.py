import re
from transformers import AutoTokenizer

class DataPreprocessor():
    def __init__(self, input_text = None):
        self.dummydata = [
            ["Cold, Sneeze, Cough", [1, 2, 3]],
            ["Stomach Hurt, Headache", [4, 5, 6]],
            ["Low blood pressure, high heartbeatrate", [2, 5, 8]],
            ["Fainting, low oxygen", [1, 8, 9]],
            ["Hurting leg, fainting, high thirst", [4, 7, 8]],
            ["Nausea, Vomiting", [0, 6, 9]],
            ["High fever, muscle pain", [2, 4, 9]],
            ["Chest pain, shortness of breath", [1, 5, 7]],
            ["Dizziness, lightheadedness", [3, 6, 8]],
            ["Joint pain, fatigue", [5, 6, 7]],
            ["Rash, itching", [4, 8, 9]],
            ["Earache, fever", [1, 2, 5]],
            ["Back pain, discomfort", [3, 4, 6]],
            ["Dry throat, difficulty swallowing", [2, 5, 9]],
            ["Weight loss, increased thirst", [1, 7, 8]],
            ["Mood swings, insomnia", [3, 5, 9]],
            ["Sore throat, fatigue", [2, 6, 9]],
            ["Swelling, bruising", [1, 3, 8]],
            ["Blurred vision, headache", [4, 5, 9]],
            ["Cold extremities, shivering", [2, 6, 8]]
        ]
        self.dummydata = self.dummydata *50
        # create empty dict with labels
        self.patient_data = {}
        self.input_text = input_text

    def trim_patient_description(self, input_text = None):
        description = self.input_text if input_text is None else input_text

        # Extract age
        age_match = re.search(r'\b(\d{1,3})[- ]?(year[- ]old|years? old)\b', description, re.IGNORECASE)
        age = age_match.group(1) + " years old" if age_match else "Age not specified"

        # Extract gender
        gender_match = re.search(r'\b(male|female|man|woman)\b', description, re.IGNORECASE)
        gender = gender_match.group(1).capitalize() if gender_match else "Gender not specified"

        # Define conditions and symptoms keywords
        condition_keywords = ["COVID-19", "pneumonia", "diabetes", "hypertension", "stroke", "cancer", "infection", "fracture"]
        symptom_keywords = ["fever", "fatigue", "pain", "cough", "shortness of breath", "nausea", "vomiting", "dizziness"]

        # Extract conditions and symptoms
        conditions = [kw for kw in condition_keywords if re.search(rf'\b{kw}\b', description, re.IGNORECASE)]
        symptoms = [kw for kw in symptom_keywords if re.search(rf'\b{kw}\b', description, re.IGNORECASE)]

        # Extract durations and time references
        time_matches = re.findall(r'\b\d+\s+(day|week|month|year)[s]?\b', description, re.IGNORECASE)
        duration = ", ".join(time_matches) if time_matches else "No time details"

        # Combine extracted details into a summary
        conditions_text = ", ".join(conditions) if conditions else "No specific conditions mentioned"
        symptoms_text = ", ".join(symptoms) if symptoms else "No specific symptoms mentioned"
        summary = f"{age}, {gender}. Conditions: {conditions_text}. Symptoms: {symptoms_text}. Duration: {duration}."

        # add to dataframe
        self.patient_data = {'Age': age, 'Gender': gender, 'Conditions': conditions_text, 'Symptoms': symptoms_text, 'Duration': duration, 'Description Summary': summary}

    def tokenize_data(self):
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        symptoms = self.patient_data['Symptoms']
        tokenized_inputs = tokenizer(symptoms, padding='max_length', max_length=50, return_tensors='pt')
        return tokenized_inputs

    def summarize_text_with_format(self, list_of_papers):
        _prompt = f"""
        Summarize papers in the following format: 
        1. Title: <title>
        2. Summary: <summary>
        
        These are the papers:
        
"""

        # TODO: add link to paper
        for paper in list_of_papers:
            _prompt += f"""{paper}\n"""

        return _prompt

    def get_dummy_data(self):
        return self.dummydata

    def get_dummy_data_at_index(self, index):
        return self.dummydata[index]

    def get_dummy_data_length(self):
        return len(self.dummydata)