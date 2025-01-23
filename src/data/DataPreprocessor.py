import re
from transformers import AutoTokenizer
from src.data.deep_symptom_extraction import SymptomExtractor
symptom_extractor = SymptomExtractor()

class DataPreprocessor():
    def __init__(self, symptoms = None):
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
        self.symptoms = symptoms

    def tokenize_data(self):
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        # TODO: get list of symptoms
        symptoms = self.symptoms
        tokenized_inputs = tokenizer(symptoms, padding='max_length', max_length=1000, return_tensors='pt')
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