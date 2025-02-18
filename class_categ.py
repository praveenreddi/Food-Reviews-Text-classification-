def classify_comment_rationale(user_comment):
    try:
        print(f"\nProcessing text:\n{user_comment}\n")
        system_message = get_system_message()
        messages = get_prompt_message(system_message, user_comment)

        response_text = llm._call(messages)
        print(f"Response received:\n{response_text}\n")

        if not response_text:
            print(f"Empty response for text:\n{user_comment}\n")
            return ("Error", "Error", "Error", 0)

        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                print(f"No JSON found in response:\n{response_text}\n")
                return ("Error", "Error", "Error", 0)

            json_str = response_text[start_idx:end_idx]
            response_json = json.loads(json_str)

            return (
                response_json.get("EN_text", ""),
                response_json.get("language_code", ""),
                response_json.get("predicted_label", "unknown/vague"),
                float(response_json.get("confidence_score", 0.5))
            )

        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error for text:\n{user_comment}\n")
            print(f"Error: {str(e)}\n")
            return ("Error", "Error", "Error", 0)

    except Exception as e:
        print(f"Classification error for text:\n{user_comment}\n")
        print(f"Error: {str(e)}\n")
        return ("Error", "Error", "Error", 0)

def process_chunk(chunk_df):
    chunk_results = {}
    for idx, text in chunk_df.items():
        try:
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                chunk_results[idx] = (None, None, None, None)
            else:
                print(f"\nProcessing index {idx}")
                chunk_results[idx] = classify_comment_rationale(str(text).strip())
        except Exception as e:
            print(f"Error processing text at index {idx}:\n{text}\n")
            print(f"Error: {str(e)}\n")
            chunk_results[idx] = ("Error", "Error", "Error", 0)
    return chunk_results

# Main processing code
valid_data = valid_data.copy()  # Create a copy to avoid SettingWithCopy warning
results_dict = {}

with ThreadPoolExecutor(max_workers=10) as executor:
    threadPool_feedbacks = {}

    # Submit all chunks for processing
    for chunk_indices in tqdm(chunks):
        chunk_df = valid_comments.loc[chunk_indices]
        feedback = executor.submit(process_chunk, chunk_df)
        threadPool_feedbacks[feedback] = chunk_indices

    # Process results as they complete
    for task in tqdm(as_completed(threadPool_feedbacks),
                    total=len(threadPool_feedbacks),
                    desc="Processing Chunks"):
        try:
            chunk_results = task.result()
            results_dict.update(chunk_results)
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")

# Update the DataFrame with results
for idx, result_tuple in results_dict.items():
    try:
        valid_data.loc[idx, result_columns] = result_tuple
    except Exception as e:
        print(f"Error updating DataFrame at index {idx}: {str(e)}")
