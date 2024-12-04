# medLLM: Fine-Tuning a Medical Language Model for Clinical Decision Support

## Project Description
This project aims to fine-tune a Large Language Model (LLM), such as Llama-2, specifically for clinical decision support. The model will be trained to assist healthcare professionals in diagnosing diseases based on patient symptoms and medical history. By leveraging a curated medical dataset, the model will learn to generate recommendations and insights that can aid in clinical decision-making.

## Objectives
- Fine-tune the LLM on a curated medical dataset.
- Evaluate the model's performance in generating accurate clinical recommendations.
- Implement a user-friendly interface for healthcare professionals to interact with the model.

## Dataset Options

1. EHR-DS-QA
   - A synthetic question-and-answer dataset derived from medical discharge summaries.
   - Contains: 21,466 medical discharge summaries and 156,599 synthetically generated Q&A pairs.
   - Link to EHR-DS-QA Dataset

2. PMC-Patients
   - A large-scale dataset consisting of 167,000 patient summaries.
   - Includes: 3.1 million patient-article relevance annotations and 293,000 patient-patient similarity annotations.
   - Link to PMC-Patients Dataset

3. BioRead
   - A biomedical machine reading comprehension dataset with approximately 16.4 million passage-question instances.
   - Includes a smaller subset, BioReadLite, for those with limited computational resources.
   - Link to BioRead Dataset

4. CliCR
   - A dataset specifically designed for machine comprehension in the medical domain, containing about 100,000 queries based on BMJ Case Reports.
   - Link to CliCR Dataset

5. HealthQA
   - A consumer health question answering dataset that includes various health-related questions and answers.
   - Link to HealthQA Dataset

6. CheXpert
   - A large dataset of chest radiographs with labels for various conditions, useful for integrating visual data into patient query systems.
   - Link to CheXpert Dataset

## Model Setup
- Use Hugging Face's Transformers library to load the pre-trained Llama-2 model.
- Implement parameter-efficient fine-tuning techniques (e.g., LoRA) to adapt the model to the medical domain.

## Training
- Define hyperparameters such as batch size, learning rate, and number of epochs.
- Train the model on the prepared dataset while monitoring performance metrics like accuracy and loss.

## Evaluation
- Test the model on a separate validation set to assess its ability to generate accurate recommendations.
- Use relevant and appropriate performance metrics for evaluation.

## Deployment
- Create a simple web application API using Flask or FastAPI, or a user interface via Streamlit, where users can input patient data queries and receive recommendations from the model.

## Libraries
- Hugging Face Transformers
- Pandas
- Scikit-learn
- Flask or Streamlit for web deployment

## Note
- No UI-based platform like Unsloth should be used.
- Llama model should be uploaded to the Ollama Model Hub.
