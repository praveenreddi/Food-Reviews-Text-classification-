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

def process_chunk(chunk_df):
    chunk_results = {}
    for column in chunk_df.columns:  # Process each column in the chunk
        chunk_results[column] = {}
        for idx, text in chunk_df[column].items():
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                chunk_results[column][idx] = (None, None, None)
            else:
                chunk_results[column][idx] = classify_comment_rationale(str(text))
    return chunk_results

# Main processing code
csv_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/..."
decoded_sas_token = base64.b64decode(sas_token).decode()

# Define all columns to process
columns_to_process = ["NAVIGATION_OE", "SPEED_OE", "RELIABILITY_OE"]  # Add all your columns here

combined_data = pd.read_csv(csv_path, storage_options={"sas_token": decoded_sas_token})

# Create result columns for each input column
result_columns = {}
for column in columns_to_process:
    result_columns[column] = [
        f"emotion_summary_{column}",
        f"emotion_confidence_{column}"
    ]
    # Initialize new columns
    for result_col in result_columns[column]:
        combined_data[result_col] = None

# Process valid data for all columns
valid_masks = {col: combined_data[col].notna() for col in columns_to_process}
valid_data = combined_data[columns_to_process].loc[valid_masks[columns_to_process[0]]]
print(len(combined_data))
print(len(valid_data))

def split_into_chunks(data, chunk_size=100):
    all_indices = data.index.tolist()
    chunk_list = []
    for i in range(0, len(all_indices), chunk_size):
        chunk_indices = all_indices[i:i+chunk_size]
        chunk_list.append(chunk_indices)
    return chunk_list

CHUNK_SIZE = 100
tasks = split_into_chunks(valid_data, chunk_size=CHUNK_SIZE)
print(f"Split data into {len(tasks)} chunks")

results_dict = {col: {} for col in columns_to_process}

with ThreadPoolExecutor(max_workers=8) as executor:
    future_to_chunk = {}
    for i, chunk_indices in enumerate(tasks):
        chunk_df = valid_data.loc[chunk_indices]
        future = executor.submit(process_chunk, chunk_df)
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
            combined_data.loc[idx, result_columns[col]] = result_tuple[:2]  # Taking summary and confidence

print("Processing completed")
print(f"Total rows processed: {len(results_dict[columns_to_process[0]])}")
for col in columns_to_process:
    print(f"\nResults for {col}:")
    print(combined_data[[col] + result_columns[col]].head())

output_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/csat_emotions_thread_all_columns.csv"
combined_data.to_csv(output_path, index=False, storage_options={"sas_token": decoded_sas_token})
print("Output saved to: ", output_path)
