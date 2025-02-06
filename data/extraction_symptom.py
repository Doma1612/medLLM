import spacy
import pandas as pd
import scispacy
from scispacy.linking import EntityLinker
from spacy import displacy

# Load the SciSpaCy model for medical text
nlp = spacy.load("en_core_sci_sm")

# Example text (Symptoms in medical records)
text = "The patient shows signs of fever, continuous coughing, and shortness of breath."

# List of symptoms to match
symptom_list = [
    "fever", "cough", "shortness of breath", "headache", "fatigue", 
    "nausea", "vomiting", "sore throat", "chills", "dizziness", 
    "chest pain", "muscle aches", "loss of appetite", "sweating", 
    "runny nose", "sneezing", "congestion", "body aches", "joint pain",
    "sore muscles", "diarrhea", "abdominal pain", "rash", "skin redness", 
    "swelling", "coughing up blood", "wheezing", "blurry vision", 
    "insomnia", "irritability", "confusion", "numbness", "tingling", 
    "hearing loss", "difficulty swallowing", "hoarseness", "fatigue", 
    "hypertension", "bradycardia", "tachycardia", "hypotension", 
    "hyperglycemia", "hypoglycemia", "seizures", "syncope", 
    "palpitations", "lightheadedness", "chest tightness", 
    "wheezing", "flank pain", "hematuria", "oliguria", "anuria", 
    "urinary retention", "incontinence", "elevated heart rate", 
    "rapid breathing", "persistent cough", "night sweats", 
    "bleeding gums", "jaundice", "hepatomegaly", "splenomegaly", 
    "back pain", "visual disturbances", "weight loss", "fever spikes", 
    "delirium", "restlessness", "tingling in hands", "weakness", 
    "difficulty breathing", "swollen lymph nodes", "nausea with vomiting", 
    "constant hunger", "polydipsia", "hearing ringing", "poor circulation", 
    "cold hands and feet", "confusion", "visual floaters", "sensitivity to light", 
    "difficulty walking", "fatigue after exertion", "unexplained bruising", 
    "bone pain", "skin ulcer", "fainting", "sore throat with swollen lymph nodes", 
    "sensitivity to cold", "pins and needles", "thickened skin", 
    "dark-colored urine", "increased thirst", "dry mouth", 
    "red eyes", "bloody stools", "chronic cough", "persistent fatigue", 
    "yellowing of skin", "swollen ankles", "numbness in feet", "chronic fatigue", 
    "palpitations with dizziness", "swollen abdomen", "tremors", "muscle weakness", 
    "spitting blood", "persistent fever", "low blood pressure", "wheezing with coughing", 
    "irregular heartbeat", "loss of consciousness", "abnormal breathing patterns", 
    "confusion and disorientation", "blurred vision", "sudden weight loss", 
    "unexplained fever", "fatigue after meals", "stomach cramps", "digestive problems", 
    "gastrointestinal bleeding", "slow heart rate", "rash with itching", "ringing in ears"
]


# Function to extract symptoms from the text
def extract_symptoms(text, symptom_list):
    doc = nlp(text)
    extracted_symptoms = []

    # Iterate through the entities found by the model and check if they match a symptom
    for ent in doc.ents:
        if ent.text.lower() in symptom_list:
            extracted_symptoms.append(ent.text.lower())

    return extracted_symptoms

# Load the dataset from CSV
df = pd.read_csv('/content/PMC-Patients-oa-9995.csv')

# Assuming 'patient' column contains the medical records. If not, change this to the correct column name.
df['extracted_symptoms'] = df['patient'].apply(lambda x: extract_symptoms(x, symptom_list))

# Save the results to a new CSV file
df.to_csv('extracted_symptoms.csv', index=False)

# Display the extracted symptoms
print(df[['patient_uid', 'extracted_symptoms']])  
