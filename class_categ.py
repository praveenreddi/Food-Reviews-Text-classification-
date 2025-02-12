import pandas as pd
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_system_message_modified():
    # Define your system message here
    return """Your system message here"""

def build_prompt_message_modified(system_message, user_comment):
    # Build your prompt message here
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_comment}
    ]

def classify_comment_rationale(user_comment):
    system_message = get_system_message_modified()
    messages = build_prompt_message_modified(system_message, user_comment)
    
    try:
        completion = llm._call(messages)
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

def process_single_column(combined_data, selected_column, decoded_sas_token):
    # Set up result columns for current column
    result_columns = [
        f"emotion_summary_{selected_column}",
        f"emotion_rationale_{selected_column}",
        f"emotion_confidence_{selected_column}"
    ]
    
    for col in result_columns:
        combined_data[col] = None
    
    # Process valid data for current column
    valid_mask = combined_data[selected_column].notna()
    valid_data = combined_data.loc[valid_mask, selected_column]
    print(f"Processing {selected_column}")
    print(f"Total rows: {len(combined_data)}")
    print(f"Valid rows: {len(valid_data)}")
    
    # Process in chunks
    CHUNK_SIZE = 100
    tasks = split_into_chunks(valid_data, chunk_size=CHUNK_SIZE)
    print(f"Split data into {len(tasks)} chunks")
    
    results_dict = {}
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
                results_dict.update(chunk_results)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
    
    # Update results in combined_data
    for idx, result_tuple in results_dict.items():
        combined_data.loc[idx, result_columns] = result_tuple
    
    print(f"Processing completed for {selected_column}")
    print(f"Total rows processed: {len(results_dict)}")
    print(combined_data[[selected_column] + result_columns].head())
    
    return combined_data

def main():
    # Define paths and configurations
    columns_to_process = ["NAVIGATION_OE", "LOGIN_OE", "IMPROVEMENT_OE", "PROFILE_OE"]
    csv_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/csat_forsee.csv"
    
    try:
        # Get your sas_token from your environment or configuration
        decoded_sas_token = base64.b64decode(sas_token).decode()
        
        # Initial data load
        combined_data = pd.read_csv(csv_path, storage_options={"sas_token": decoded_sas_token})
        
        for column in columns_to_process:
            # Process current column
            combined_data = process_single_column(combined_data, column, decoded_sas_token)
            
            # Save intermediate results
            output_path = f"abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/csat_emotions_thread_{column}.csv"
            combined_data.to_csv(output_path, index=False, storage_options={"sas_token": decoded_sas_token})
            print(f"Output saved for {column} to: {output_path}")
            
            # Run required notebooks for token refresh
            print(f"Refreshing tokens after processing {column}")
            dbutils.notebook.run("./configs", timeout_seconds=600)
            dbutils.notebook.run("./GetAccessToken", timeout_seconds=600)
            dbutils.notebook.run("./Custom_APIs", timeout_seconds=600)
        
        print("All columns processed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()

