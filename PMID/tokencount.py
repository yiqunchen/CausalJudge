import csv
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

TOKEN_LIMIT = 30000

too_long_articles = []

with open("pmid_text_output.csv", mode="r", encoding="utf-8", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    all_rows = list(reader)

    for i, row in enumerate(all_rows):
        text = row.get("text", "")
        tokens = enc.encode(text)
        token_count = len(tokens)

        if token_count > TOKEN_LIMIT:
            too_long_articles.append({
                "row_index": i,
                "pmid": row.get("pmid", ""),
                "token_count": token_count
            })

# Report
print(f"Found {len(too_long_articles)} articles over {TOKEN_LIMIT} tokens.\n")
for article in too_long_articles:
    print(f"Row {article['row_index']+1}: PMID {article['pmid']} — {article['token_count']} tokens")
