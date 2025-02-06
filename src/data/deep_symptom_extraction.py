import spacy
import pandas as pd


from src.utils.constants import symptom_list


class SymptomExtractor:
    def __init__(self):
        # Load the SciSpaCy model for medical text
        self.nlp = spacy.load("en_core_sci_sm")
        # List of symptoms to match

        # Convert the list of symptoms to a set for fast lookup
        self.symptom_set = {symptom.lower() for symptom in symptom_list}
    
    def extract_symptoms(self, text):
        # Extract symptoms from the provided text
        return [
            word.text for word in self.nlp(text).ents 
        ]

if __name__=="__main__":

#    Example usage:
    # Create an instance of SymptomExtractor with your symptom list
    symptom_extractor = SymptomExtractor(symptom_list)

    # Example input text for the extraction
    text = "The patient shows signs of fever, continuous coughing, and shortness of breath."

    # Extract symptoms from the text
    extracted_symptoms = symptom_extractor.extract_symptoms(text)
    print("Extracted Symptoms:", extracted_symptoms)