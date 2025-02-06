import spacy
from src.data.deep_symptom_extraction import SymptomExtractor
symptom_extractor = SymptomExtractor()

class DataPreprocessor():
    def __init__(self, symptoms = None):
        # Load the SciSpaCy model for medical text
        self.nlp = spacy.load("en_core_sci_sm")

    def summarize_text_with_format(self, paper):
        _prompt = """
        Summarize this paper in the following format:
        1. Title: <title>

        2. Summary: <summary>

        This is the paper:

        """

        _prompt += f"""{paper}\n"""

        return _prompt

    def extract_symptoms(self, text):
        # Extract symptoms from the provided text
        # text = "The patient shows signs of fever, continuous coughing, and shortness of breath."
        return [word.text for word in self.nlp(text).ents]
