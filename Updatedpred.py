def get_system_message_modified():
    delimiter = "####"
    # Define both label and emotion categories - use exactly these lists
    prediction_labels = [
        "unable_to_login", "navigation_issues", "long_wait_times_to_connect_with_agent"
        # Add other prediction labels as needed
    ]
    emotion_labels = [
        "Happy", "Sad", "Angry", "Frustration",
        "Sarcastic", "Neutral", "Surprised",
        "Fear", "Unhappy", "Unknown"
    ]

    labels_str = ", ".join(prediction_labels)
    emotions_str = ", ".join(emotion_labels)

    system_message = f"""You are an advanced AI assistant analyzing user queries and user queries will be delimited with {delimiter} characters.
Classify each query into ONLY one of these specific categories:
{labels_str}. If you cannot classify the query, return "unknown".

Also determine the emotional tone of the comment from ONLY these emotions:
{emotions_str}. Do not create new emotion categories.

Return with a JSON object containing:
{{
    "predicted_label": one of the specified labels listed above or "unknown",
    "emotion_summary": one of the specified emotions listed above,
    "emotion_confidence": confidence score between 0 and 1
}}

IMPORTANT: Do not include the delimiter '{delimiter}' in your response. Return only the JSON object.
IMPORTANT: Only use the exact labels and emotions provided in the lists above.
""".strip()

    return system_message

def build_prompt_message_modified(system_message, user_comment):
    delimiter = "####"
    return [
        {"role":"system", "content":system_message},
        {"role":"user", "content":f"{delimiter}{user_comment}{delimiter}"}
    ]

def classify_comment_rationale(user_comment):
    system_message = get_system_message_modified()
    messages = build_prompt_message_modified(system_message, user_comment)

    try:
        completion = llm_call(messages)

        try:
            response_json = json.loads(completion)

            # Validate that prediction is from allowed list
            prediction_labels = [
                "unable_to_login", "navigation_issues", "long_wait_times_to_connect_with_agent", "unknown"
                # Add other prediction labels as needed
            ]
            emotion_labels = [
                "Happy", "Sad", "Angry", "Frustration",
                "Sarcastic", "Neutral", "Surprised",
                "Fear", "Unhappy", "Unknown"
            ]

            prediction = response_json.get("predicted_label", "unknown")
            if prediction not in prediction_labels:
                print(f"Invalid prediction label: {prediction}")
                prediction = "unknown"

            emotion = response_json.get("emotion_summary", "Neutral")
            if emotion not in emotion_labels:
                print(f"Invalid emotion: {emotion}")
                emotion = "Neutral"

            confidence = float(response_json.get("emotion_confidence", 0.5))

            return (prediction, emotion, confidence)

        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            print(f"Problematic text:{completion}")
            return ("unknown", "Neutral", 0.5)

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return ("unknown", "Neutral", 0.5)

def process_chunk(chunk_df):
    chunk_results = {}

    for idx, text in chunk_df.items():
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            chunk_results[idx] = (None, None, None)
        else:
            chunk_results[idx] = classify_comment_rationale(str(text))

    return chunk_results

def split_into_chunks(data, chunk_size=100):
    all_indices = data.index.tolist()
    chunk_list = []

    for i in range(0, len(all_indices), chunk_size):
        chunk_indices = all_indices[i:i+chunk_size]
        chunk_list.append(chunk_indices)

    return chunk_list

def file_exists(file_path, storage_options):
    """Check if a file exists in Azure storage"""
    try:
        # Using dbutils to check if file exists
        return dbutils.fs.ls(file_path) is not None
    except:
        return False

# Main processing code
import datetime

# Get current date for tracking
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

csv_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/csat_forsee.csv"
output_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/csat_emotions_labels_all_columns_predictions.csv"
decoded_sas_token = base64.b64decode(sas_token).decode()
storage_options = {"sas_token": decoded_sas_token}

# List of columns to process
columns_to_process = ["LOGIN_OE", "NAVIGATION_OE", "IMPROVEMENT_OE", "PROFILE_OE"]

# Load the input data
input_data = pd.read_csv(csv_path, storage_options=storage_options)
print(f"Total rows in input dataset: {len(input_data)}")

# Add a unique identifier column if it doesn't exist
if 'unique_id' not in input_data.columns:
    input_data['unique_id'] = input_data.index.astype(str) + '_' + current_date

# Check if output file exists and load it
existing_output = None
try:
    existing_output = pd.read_csv(output_path, storage_options=storage_options)
    print(f"Loaded existing output file with {len(existing_output)} rows")
except:
    print("No existing output file found, will create a new one")

# Create a new dataframe for today's results
today_results = input_data.copy()
today_results['run_date'] = current_date

# Create all result columns upfront
for selected_column in columns_to_process:
    result_columns = [
        f"predicted_label_{selected_column}",
        f"emotion_summary_{selected_column}",
        f"emotion_confidence_{selected_column}"
    ]
    for col in result_columns:
        if col not in today_results.columns:
            today_results[col] = None

# Process each column
for selected_column in columns_to_process:
    print(f"\nProcessing column: {selected_column}")

    # Define result columns for this column
    result_columns = [
        f"predicted_label_{selected_column}",
        f"emotion_summary_{selected_column}",
        f"emotion_confidence_{selected_column}"
    ]

    # Filter valid data
    valid_mask = today_results[selected_column].notna()
    valid_data = today_results.loc[valid_mask, selected_column]
    print(f"Valid rows to process: {len(valid_data)}")

    if len(valid_data) == 0:
        print(f"No valid data in column {selected_column}, skipping")
        continue

    # Process in chunks
    CHUNK_SIZE = 100
    tasks = split_into_chunks(valid_data, chunk_size=CHUNK_SIZE)
    print(f"Split data into {len(tasks)} chunks")

    results_dict = {}

    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_chunk = {}
        for i, chunk_indices in enumerate(tasks):
            chunk_df = valid_data.loc[chunk_indices]
            future = executor.submit(process_chunk, chunk_df)
            future_to_chunk[future] = chunk_indices

        for future in tqdm(as_completed(future_to_chunk), total=len(future_to_chunk), desc=f"Processing {selected_column}"):
            chunk_indices = future_to_chunk[future]
            try:
                chunk_results = future.result()
                results_dict.update(chunk_results)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")

    # Update the dataframe with results
    for idx, result_tuple in results_dict.items():
        if result_tuple[0] is not None:  # Only update if we got valid results
            today_results.loc[idx, result_columns] = result_tuple

    print(f"Processing completed for {selected_column}")
    print(f"Total rows processed: {len(results_dict)}")

# Combine today's results with existing results (if any)
if existing_output is not None:
    # Append today's results to existing output
    combined_output = pd.concat([existing_output, today_results], ignore_index=True)
    print(f"Combined output has {len(combined_output)} rows")
else:
    combined_output = today_results
    print(f"New output has {len(combined_output)} rows")

# Save the combined results
combined_output.to_csv(output_path, index=False, storage_options=storage_options)
print("\nResults appended to: ", output_path)
print("Success")
