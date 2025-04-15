import json
import pandas as pd
import os

# Define the path to your JSONL file
input_file_path = r'RAG_Project\data\arxiv-metadata-oai-snapshot.json'
output_file_path = r'RAG_Project\data\cleaned_arxiv_metadata.csv'

# Check if the input file exists
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"The file {input_file_path} does not exist.")

# Initialize a list to store the extracted records
records = []

# Define the maximum number of records to process (adjust as needed)
max_records = 1000

# Open the JSONL file and process line by line
with open(input_file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i >= max_records:
            break
        try:
            data = json.loads(line)
            # Extract relevant fields
            record = {
                'id': data.get('id'),
                'title': data.get('title'),
                'abstract': data.get('abstract'),
                'categories': data.get('categories'),
                'update_date': data.get('update_date')
            }
            records.append(record)
        except json.JSONDecodeError:
            continue

# Create a DataFrame from the records
df = pd.DataFrame(records)

# Display the first few entries
print(df.head())

# Save the cleaned data to a CSV file
df.to_csv(output_file_path, index=False)
print(f"Processed data saved to {output_file_path}")
