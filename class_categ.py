import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import pandas as pd
import base64

# Initial setup
%run ./config_files/configs
%run -/GetAccess Token
%run - /custom_APIs

class TokenManager:
    def __init__(self):
        self.last_refresh_time = time.time()
        self.refresh_interval = 3500  # 58 minutes in seconds
        self.lock = threading.Lock()

    def refresh_if_needed(self):
        current_time = time.time()
        with self.lock:
            if (current_time - self.last_refresh_time) >= self.refresh_interval:
                print(f"\nRefreshing token at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                try:
                    # Run the token refresh notebooks
                    dbutils.notebook.run("./config_files/configs", timeout_seconds=300)
                    dbutils.notebook.run("./GetAccess Token", timeout_seconds=300)
                    dbutils.notebook.run("./custom_APIs", timeout_seconds=300)

                    # Reinitialize the LLM
                    global llm
                    llm = LlamaLLM(llm_model="llama")

                    self.last_refresh_time = time.time()
                    print("Token refresh successful")
                except Exception as e:
                    print(f"Error refreshing token: {str(e)}")
                    raise

# Initialize LLM and TokenManager
llm = LlamaLLM(llm_model="llama")
token_manager = TokenManager()

def get_system_message_modified():
    system_message = """Analyze the sentiment and emotion in the following comment.
    Classify the emotion into one of these specific categories:
    Happy, Sad, Angry, Frustration, Sarcastic, Neutral, Surprised, Fear, Unhappy, or Unknown.
    Respond with a JSON object containing:
    "emotion_summary": one of the specified emotions listed above,
    "emotion_confidence": confidence score between 0 and 1

    Guidelines for classification:
    - Happy: expressions of joy, satisfaction, excitement, positive feelings
    - Sad: expressions of sadness, disappointment
    - Angry: expressions of anger, rage
    - Frustration: expressions of frustration, annoyance
    - Sarcastic: expressions of sarcasm, irony
    - Neutral: no clear emotion, objective statements
    - Surprised: expressions of surprise, shock, amazement
    - Fear: expressions of fear, worry, anxiety
    - Unhappy: expressions of general unhappiness, discontent
    - Unknown: use when the emotion is unclear, ambiguous, or cannot be determined with confidence

    Important:
    - Always respond with exactly one of these 10 emotions: Happy, Sad, Angry, Frustration, Sarcastic, Neutral, Surprised, Fear, Unhappy, Unknown
    - Use 'Unknown' when you're not confident about the emotion or when the text is unclear
    - If confidence is below 0.4, default to 'Unknown'

    Do not use any other emotion categories"""
    return system_message

def build_prompt_message_modified(system_message, user_comment):
    delimiter = "####"
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{user_comment}"}
    ]

def classify_comment_rationale(user_comment):
    # Check and refresh token if needed
    token_manager.refresh_if_needed()

    system_message = get_system_message_modified()
    messages = build_prompt_message_modified(system_message, user_comment)
    try:
        completion = llm_call(messages)

        try:
            response_json = json.loads(completion)
            emotion = response_json.get("emotion_summary", "neutral")
            confidence = float(response_json.get("emotion_confidence", 0.5))

            if emotion is None:
                print("No emotion_summary in response")
                return ("neutral", 0.5)
            if confidence is None:
                print("No emotion_confidence in response")
                confidence = 0.5
            else:
                confidence = float(confidence)
            return (emotion, confidence)

        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            print(f"Problematic text: {completion}")
            return ("neutral", 0.5)

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return ("neutral", 0.5)

def process_chunk(chunk_df):
    chunk_results = {}
    for column in chunk_df.columns:
        chunk_results[column] = {}
        for idx, text in chunk_df[column].items():
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                chunk_results[column][idx] = (None, None)
            else:
                chunk_results[column][idx] = classify_comment_rationale(str(text))
    return chunk_results

def split_into_chunks(data, chunk_size=100):
    all_indices = data.index.tolist()
    chunk_list = []
    for i in range(0, len(all_indices), chunk_size):
        chunk_indices = all_indices[i:i+chunk_size]
        chunk_list.append(chunk_indices)
    return chunk_list

def main():
    try:
        # Data loading and preparation
        csv_path = "abfss://opsdashboards@blackbirdprodflatastore.dfs.core.windows.net/VOC/pitch26/CSAT/csat_forsee.csv"
        decoded_sas_token = base64.b64decode(sas_token).decode()
        columns_to_process = ["IMPROVEMENT_OE", "LOGIN_OE", "PROFILE_OE", "NAVIGATION_OE"]

        # Read data
        combined_data = pd.read_csv(csv_path, storage_options={"sas_token": decoded_sas_token})

        # Prepare result columns
        result_columns = {}
        for column in columns_to_process:
            result_columns[column] = [
                f"emotion_summary_{column}",
                f"emotion_confidence_{column}"
            ]
            for result_col in result_columns[column]:
                combined_data[result_col] = None

        # Prepare valid data
        valid_masks = {col: combined_data[col].notna() for col in columns_to_process}
        valid_data = combined_data[columns_to_process].loc[valid_masks[columns_to_process[0]]]

        # Split into chunks
        CHUNK_SIZE = 100
        tasks = split_into_chunks(valid_data, chunk_size=CHUNK_SIZE)
        print(f"Split data into {len(tasks)} chunks")

        # Process chunks in parallel
        results_dict = {col: {} for col in columns_to_process}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_chunk = {}

            # Submit all chunks
            for i, chunk_indices in enumerate(tasks):
                chunk_df = valid_data.loc[chunk_indices]
                future = executor.submit(process_chunk, chunk_df)
                future_to_chunk[future] = chunk_indices

            # Process results as they complete
            for future in tqdm(as_completed(future_to_chunk), total=len(future_to_chunk), desc="Processing chunks"):
                chunk_indices = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    for col in columns_to_process:
                        results_dict[col].update(chunk_results[col])
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")

        # Update final results
        for col in columns_to_process:
            for idx, result_tuple in results_dict[col].items():
                if result_tuple:
                    combined_data.loc[idx, result_columns[col]] = result_tuple

        print("Processing completed")
        print(f"Total rows processed: {len(results_dict[columns_to_process[0]])}")

        # Print sample results
        for col in columns_to_process:
            print(f"\nResults for {col}:")
            print(combined_data[[col] + result_columns[col]].head())

        # Save final results
        output_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/csat_emotions_predictions_final.csv"
        combined_data.to_csv(output_path, index=False, storage_options={"sas_token": decoded_sas_token})

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
