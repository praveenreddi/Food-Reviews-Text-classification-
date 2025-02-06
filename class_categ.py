import pandas as pd
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_system_message_modified():
    # Add your system message logic here
    return "Your system message"

def build_prompt_message_modified(system_message, user_comment):
    # Add your prompt building logic here
    return []

def llm_call(messages):
    # Add your LLM call logic here
    return []

def classify_comment_rationale(user_comment):
    system_message = get_system_message_modified()
    messages = build_prompt_message_modified(system_message, user_comment)

    try:
        completion = llm_call(messages)
        response_text = ""

        if isinstance(completion, list) and len(completion) > 0:
            response_text = completion[0].message.content.strip()
        elif hasattr(completion, "choices") and completion.choices:
            response_text = completion.choices[0].message.content.strip()

        if not response_text:
            return ("neutral", "Error parsing", 0.5)

        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            return ("neutral", "Error parsing", 0.5)

        return (
            response_json.get("emotion_summary", "neutral"),
            response_json.get("emotion_rationale", "No Rationale"),
            float(response_json.get("emotion_confidence", 0.5))
        )

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return ("neutral", "Error parsing", 0.5)

def process_chunk(chunk_df, columns_to_process):
    chunk_results = {}
    for column in columns_to_process:
        chunk_results[column] = {}
        for idx, text in chunk_df[column].items():
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                chunk_results[column][idx] = (None, None, None)
            else:
                chunk_results[column][idx] = classify_comment_rationale(str(text))
    return chunk_results

def split_into_chunks(data, chunk_size=100):
    all_indices = data.index.tolist()
    return [all_indices[i:i+chunk_size] for i in range(0, len(all_indices), chunk_size)]

def main():
    # Azure Storage path and configuration
    csv_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/your_input_file.csv"
    sas_token = "your_sas_token"  # Replace with your actual SAS token
    decoded_sas_token = base64.b64decode(sas_token).decode()

    # Define columns to process
    columns_to_process = ["NAVIGATION_OE", "SPEED_OE", "RELIABILITY_OE"]  # Add all columns you want to process

    # Read the data
    combined_data = pd.read_csv(csv_path, storage_options={"sas_token": decoded_sas_token})

    # Create result columns for each input column
    result_columns = {}
    for col in columns_to_process:
        result_columns[col] = [
            f"emotion_summary_{col}",
            f"emotion_confidence_{col}"
        ]
        # Initialize new columns
        for result_col in result_columns[col]:
            combined_data[result_col] = None

    # Create valid mask for all columns
    valid_masks = {col: combined_data[col].notna() for col in columns_to_process}
    valid_data = combined_data[columns_to_process].loc[valid_masks[columns_to_process[0]]]

    print(f"Total rows in dataset: {len(combined_data)}")
    print(f"Valid rows for processing: {len(valid_data)}")

    # Split data into chunks
    CHUNK_SIZE = 100
    tasks = split_into_chunks(valid_data, chunk_size=CHUNK_SIZE)
    print(f"Split data into {len(tasks)} chunks")

    # Process chunks using ThreadPoolExecutor
    results_dict = {col: {} for col in columns_to_process}

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_chunk = {}
        for i, chunk_indices in enumerate(tasks):
            chunk_df = valid_data.loc[chunk_indices]
            future = executor.submit(process_chunk, chunk_df, columns_to_process)
            future_to_chunk[future] = chunk_indices

        for future in tqdm(as_completed(future_to_chunk), total=len(future_to_chunk), desc="Processing chunks"):
            chunk_indices = future_to_chunk[future]
            try:
                chunk_results = future.result()
                # Update results for each column
                for col in columns_to_process:
                    results_dict[col].update(chunk_results[col])
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")

    # Update the dataframe with results
    for col in columns_to_process:
        for idx, result_tuple in results_dict[col].items():
            if result_tuple:
                combined_data.loc[idx, result_columns[col]] = result_tuple[:2]  # Only taking summary and confidence

    print("Processing completed")
    print(f"Total columns processed: {len(columns_to_process)}")
    for col in columns_to_process:
        print(f"Total rows processed for {col}: {len(results_dict[col])}")
        print(combined_data[[col] + result_columns[col]].head())

    # Save results
    output_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/csat_emotions_thread_all_columns.csv"
    combined_data.to_csv(output_path, index=False, storage_options={"sas_token": decoded_sas_token})
    print("Output saved to: ", output_path)

if __name__ == "__main__":
    main()
