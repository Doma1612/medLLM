import pandas as pd
import numpy as np
import requests
import io
import fitz  # PyMuPDF
from tqdm import tqdm

class DataTransformer:
    """Base class for data transformation tasks."""
    
    def save_results(self, results, file_name):
        """Saves the results to a CSV file."""
        df = pd.DataFrame(results)
        df.to_csv(file_name, index=False)
        print(f"Saved results to {file_name}")


class PDFExtractor(DataTransformer):
    """Class for extracting text from PDFs and processing a DataFrame."""
    
    def extract_text_from_pdf(self, url):
        """Extracts text from a PDF given its URL."""
        if pd.isna(url):
            return None
        try:
            session = requests.Session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Referer': 'https://example.com',
            }
            response = session.get(url, headers=headers)
            if response.status_code == 200:
                pdf_file = io.BytesIO(response.content)
                pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
                text = ""
                for page in pdf_document:
                    text += page.get_text()
                pdf_document.close()
                return text
            else:
                return f"Failed to download PDF (status code: {response.status_code})"
        except Exception as e:
            return str(e)

    def process_dataframe_splits(self, df_split, batch_number, output_path_template):
        """Processes a split of the DataFrame to extract PDF texts."""
        results = []
        
        for index, row in tqdm(df_split.iterrows(), total=df_split.shape[0], desc=f"Processing batch {batch_number}"):
            full_text = self.extract_text_from_pdf(row['full_text_url'])
            results.append({'PMID': row['PMID'], 'full_text': full_text})
        
        self.save_results(results, output_path_template.format(batch_number))


def main():
    article_data_file_path = "/home/dominik/Documents/Informatik/1_Semester/medLLM/data/pmc_patients/processed/article_data_final_state.csv"
    full_text_output_path = "/home/dominik/Documents/Informatik/1_Semester/medLLM/data/pmc_patients/processed/examples/full_texts_subset_{}.csv"
    
    df = pd.read_csv(article_data_file_path)
    
    # Filter the DataFrame to only include rows where 'full_text_url' is not NaN
    df = df.dropna(subset=['full_text_url'])
    df = df.sort_values(by="PMID")
    # for testing
    # df = df.head(100)
    # Split the DataFrame into 5 approximately equal parts
    df_splits = np.array_split(df, 5)
    
    pdf_extractor = PDFExtractor()
    
    for batch_number, df_split in enumerate(df_splits, start=1):
        pdf_extractor.process_dataframe_splits(df_split, batch_number, full_text_output_path)
    
    print("Text extraction completed and saved results.")


if __name__ == "__main__":
    main()