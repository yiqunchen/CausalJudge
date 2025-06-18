import json
import pandas as pd

# Load each JSON line into a list of dictionaries
with open("PMID_all_text.jsonl", "r") as file:
    data = [json.loads(line) for line in file]

# Convert the list into a DataFrame
df = pd.DataFrame(data)

# Ensure full text is shown, not truncated
pd.set_option('display.max_colwidth', None)

# Print the column names
print("Columns:", df.columns.tolist())

# Preview the first 3 rows (full text included)
print(df.head(3))