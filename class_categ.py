import pandas as pd
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# File path and data loading
csv_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/OL_data/OL_results/services/VOCOpinionLabData_services_monthResults.xlsx"
services_data = pd.read_excel(csv_path, storage_options={"sas_token": base64.b64decode(sas_token).decode()})

# Data preprocessing
services_data['Submission Date'] = pd.to_datetime(services_data['Submission Date'])
latest_date = pd.Timestamp('2025-01-26 00:00:00')
services_data = services_data[services_data["Submission Date"] > latest_date].copy()  # Added .copy()

valid_data = services_data[~services_data["Comments"].isna()].copy()  # Added .copy()
valid_comments = valid_data["Comments"]
comments_indices = valid_comments.index.to_list()

chunk_size = 100
chunks = [comments_indices[i:i + chunk_size] for i in range(0, len(comments_indices), chunk_size)]

def get_system_message():
    delimiter = "```"
    example_output = {
        "EN_text": "translated english text",
        "language_code": "translated language code",
        "predicted_label": "",
        "confidence_score": 0
    }

    system_message = f"""
    You have a task of 2 operations.
    Operation 1:
    You will be provided with list of customer service queries in Non English language.
    The customer service query will be delimited with {delimiter} characters.
    I need you to convert the given text to English language along with the Non English Language code.
    If you cannot convert the text, respond "I cannot".
    Below are the keys to be returned in the output: EN_text, language_code.

    Operation 2:
    Classify the translated english query from the following labels with confidence score.
    If you cannot classify the query, create a new label or "unknown".
    Below are the keys to be returned in the output: predicted_label, confidence_score.

    Final output:
    Final output response should be in below json format:
    {json.dumps(example_output)}
    """
    return system_message

def get_prompt_message(system_message, test_query):
    delimiter = "```"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{delimiter}{str(test_query)}{delimiter}"}
    ]
    return messages

def classify_comment_rationale(user_comment):
    try:
        system_message = get_system_message()
        messages = get_prompt_message(system_message, user_comment)
        response_text = llm._call(messages)

        # Parse the response
        response_json = json.loads(response_text)

        return (
            response_json.get("EN_text", ""),
            response_json.get("language_code", ""),
            response_json.get("predicted_label", "unknown/vague"),
            float(response_json.get("confidence_score", 0.5))
        )
    except Exception as e:
        print(f"Error processing comment: {str(e)}")
        return ("Error", "Error", "Error", 0)

def process_chunk(chunk_df):
    chunk_results = {}
    for idx, text in chunk_df.items():
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            chunk_results[idx] = (None, None, None, None)
        else:
            chunk_results[idx] = classify_comment_rationale(str(text))
    return chunk_results

# Process chunks with ThreadPoolExecutor
results_dict = {}
with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_chunk = {
        executor.submit(process_chunk, valid_comments.loc[chunk_indices]): chunk_indices
        for chunk_indices in chunks
    }

    for future in tqdm(as_completed(future_to_chunk), total=len(future_to_chunk), desc="Processing Chunks"):
        chunk_indices = future_to_chunk[future]
        try:
            chunk_results = future.result()
            results_dict.update(chunk_results)
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")

# Update the DataFrame
result_columns = ["translated_text", "language_code", "predicted_label", "confidence_score"]
for idx, result_tuple in results_dict.items():
    valid_data.loc[idx, result_columns] = result_tuple

# Display results
print(valid_data[result_columns].head())
