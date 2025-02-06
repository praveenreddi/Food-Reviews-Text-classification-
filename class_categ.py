import pandas as pd
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration - UPDATE THESE VALUES
SAS_TOKEN = "your_base64_encoded_sas_token_here"
TARGET_COLUMNS = ["NAVIGATION_OE", "FEEDBACK", "OTHER_COLUMN"]  # Add your columns
CSV_PATH = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/your_file.csv"
MAX_ROWS = 500  # Set to None for full dataset

def get_system_message_modified():
    return """Analyze user comments delimited by ###|#. Return EXACTLY this JSON:
{
  "emotion_summary": "happy/sad/surprised/anger/fear/disgust/sarcastic/neutral",
  "emotion_rationale": "brief explanation",
  "emotion_confidence": 0.99
}"""

def classify_comment_rationale(user_comment):
    system_message = get_system_message_modified()
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"###|#{user_comment}#|###"}
    ]

    try:
        # Replace with your actual LLM API call
        completion = llm._call(messages)  # Mock implementation

        # Extract response text
        if hasattr(completion, 'choices') and completion.choices:
            response_text = completion.choices[0].message.content.strip()
        else:
            return ("neutral", "API Error", 0.5)

        # Clean JSON response
        response_text = response_text.replace("'", '"').split('{', 1)[-1].rsplit('}', 1)[0]
        response_json = json.loads("{" + response_text + "}")

        return (
            response_json.get("emotion_summary", "neutral").lower(),
            response_json.get("emotion_rationale", "No rationale"),
            min(1.0, max(0.0, float(response_json.get("emotion_confidence", 0.5))))
        )

    except Exception as e:
        print(f"Classification error: {str(e)}")
        return ("neutral", "Processing Error", 0.5)

def process_chunk(chunk_df):
    chunk_results = {}
    for idx, row in chunk_df.iterrows():
        row_result = {}
        for col in TARGET_COLUMNS:
            text = str(row[col]) if pd.notna(row[col]) else ""
            if text.strip():
                result = classify_comment_rationale(text)
                row_result.update({
                    f"emotion_summary_{col}": result[0],
                    f"emotion_confidence_{col}": result[2]
                })
            else:
                row_result.update({
                    f"emotion_summary_{col}": None,
                    f"emotion_confidence_{col}": None
                })
        chunk_results[idx] = row_result
    return chunk_results

def main():
    # Initialize result columns
    result_columns = []
    for col in TARGET_COLUMNS:
        result_columns.extend([f"emotion_summary_{col}", f"emotion_confidence_{col}"])

    # Read data
    decoded_sas_token = base64.b64decode(SAS_TOKEN).decode()
    combined_data = pd.read_csv(
        CSV_PATH,
        storage_options={"sas_token": decoded_sas_token},
        nrows=MAX_ROWS
    )

    # Initialize columns
    for col in result_columns:
        combined_data[col] = None

    # Process data
    valid_mask = combined_data[TARGET_COLUMNS].notna().any(axis=1)
    valid_data = combined_data.loc[valid_mask, TARGET_COLUMNS]

    def split_chunks(data, chunk_size=50):
        indices = data.index.tolist()
        return [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for chunk_indices in split_chunks(valid_data):
            chunk_df = valid_data.loc[chunk_indices]
            future = executor.submit(process_chunk, chunk_df)
            futures[future] = chunk_indices

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            chunk_indices = futures[future]
            try:
                results.update(future.result())
            except Exception as e:
                print(f"Chunk error: {str(e)}")

    # Update final dataframe
    for idx, res in results.items():
        for col in TARGET_COLUMNS:
            combined_data.at[idx, f"emotion_summary_{col}"] = res.get(f"emotion_summary_{col}")
            combined_data.at[idx, f"emotion_confidence_{col}"] = res.get(f"emotion_confidence_{col}")

    # Save results
    output_path = CSV_PATH.replace(".csv", "_PROCESSED.csv")
    combined_data.to_csv(output_path, index=False, storage_options={"sas_token": decoded_sas_token})
    print(f"Processing complete. Results saved to: {output_path}")

if __name__ == "__main__":
    main()
