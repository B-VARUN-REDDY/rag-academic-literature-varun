import json
import pandas as pd

# Path to your JSONL file
file_path = r'RAG_Project\data\arxiv-metadata-oai-snapshot.json'

# Initialize a list to store selected records
records = []

# Define the maximum number of records to process (adjust as needed)
max_records = 1000

# Open the file and process line by line
with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
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
