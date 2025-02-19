def create_qa_chain():
    # Increase the number of relevant chunks retrieved
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,  # Increase this number to get more relevant chunks
            "fetch_k": 20  # Fetch more documents before selecting top k
        }
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use 'stuff' to include all retrieved documents
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# Modified prompt for better coverage
def analyze_detailed_issues(result_dict):
    prompt = """
    Analyze ALL the provided feedback thoroughly and create a comprehensive summary.
    You MUST check and include issues for ALL of these categories if they exist:

    1. Web Acct Mgmt (including):
       - Registration & Login
       - Profile
       - ViewBill & Payment
       - TOBR
       - View Usage

    2. Web Sales

    3. Support & Chat (including):
       - Online Support
       - Device Unlock
       - Outage Portal

    Format the output exactly like this:

    Web Acct Mgmt - Below are the themes observed for Web Acct Mgmt journey this week are -

    • Registration & Login - [List ALL specific issues with exact error messages] (include ALL related QM links)

    [Continue with other categories...]

    Important rules:
    1. DO NOT skip any category that has issues
    2. Include ALL QM links for each issue
    3. Check thoroughly for issues in each category
    4. Use exact error messages in quotes
    5. If a category truly has no issues, only then mark it as "No issues reported"

    Results to analyze:
    {text}
    """

    try:
        result_text = result_dict.get('result', '')
        # Ensure we're getting all the content
        if isinstance(result_text, list):
            result_text = ' '.join(result_text)

        formatting_prompt = prompt.format(text=result_text)

        # Increase max tokens if needed
        formatted_result = llm(formatting_prompt, max_tokens=4000)

        print(formatted_result)
        return formatted_result

    except Exception as e:
        print(f"Error in formatting: {e}")
        return None

# Usage with modified query
try:
    query = """
    Provide a comprehensive analysis of ALL customer feedback and issues across ALL categories:
    - Web Account Management (including all subcategories)
    - Web Sales
    - Support & Chat

    Include ALL relevant issues and their corresponding QM links.
    Do not skip any category that has reported issues.
    """

    # Increase the number of relevant chunks
    result = chain({"query": query})
    analysis = analyze_detailed_issues(result)

except Exception as e:
    print(f"Error: {e}")



import pandas as pd
import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import base64

# Reading the data
csv_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.window/-net/VOC/OL_data/OL_results/services/VOCOpinionLabData_services_monthResults.xlsx"
services_data = pd.read_excel(csv_path, storage_options={"sas_token": base64.b64decode(sas_token).decode()})

# Data preprocessing
services_data['Submission Date'] = pd.to_datetime(services_data['Submission Date'])
latest_date = pd.Timestamp('2025-01-26 00:00:00')
services_data = services_data[services_data["Submission Date"] > latest_date]

valid_data = services_data[~services_data["Comments"].isna()]
valid_comments = valid_data["Comments"]
comments_indices = valid_comments.index.to_list()

# Chunking with unique indices
chunk_size = 100
chunks = [comments_indices[i:i + chunk_size] for i in range(0, len(comments_indices), chunk_size)]

def get_system_message():
    # Your existing get_system_message implementation
    pass

def get_prompt_message(system_message, test_query):
    # Your existing get_prompt_message implementation
    pass

def classify_single_comment(text, idx):
    """Process a single comment and handle errors individually"""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return idx, ("Error - Empty Text", "Error", "Error", 0)
    
    try:
        system_message = get_system_message()
        messages = get_prompt_message(system_message, str(text))
        
        response_text = llm_call(messages)
        if not response_text:
            return idx, ("Error - No Response", "Error", "Error", 0)

        # Clean and parse response
        response_text = response_text.strip()
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > 0:
            response_text = response_text[start_idx:end_idx]
            response_text = response_text.replace('"', '"').replace("'", "'")
            
            try:
                response_json = json.loads(response_text)
            except:
                try:
                    response_dict = ast.literal_eval(response_text)
                    response_json = response_dict if isinstance(response_dict, dict) else {}
                except:
                    return idx, ("Error - Parse Failed", "Error", "Error", 0)
            
            return idx, (
                str(response_json.get("EN_text", "")),
                str(response_json.get("language_code", "")),
                str(response_json.get("predicted_label", "unknown/vague")),
                float(response_json.get("confidence_score", 0.5))
            )
        
        return idx, ("Error - Invalid JSON", "Error", "Error", 0)
    
    except Exception as e:
        return idx, (f"Error - {str(e)}", "Error", "Error", 0)

def process_chunk(chunk_indices, valid_comments):
    """Process a chunk of comments with individual error handling"""
    chunk_results = {}
    for idx in chunk_indices:
        text = valid_comments.loc[idx]
        _, result = classify_single_comment(text, idx)
        chunk_results[idx] = result
    return chunk_results

# Main processing
results_dict = {}
processed_indices = set()

with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_chunk = {
        executor.submit(process_chunk, chunk, valid_comments): chunk 
        for chunk in chunks
    }
    
    for future in tqdm(as_completed(future_to_chunk), total=len(future_to_chunk), desc="Processing chunks"):
        chunk_indices = future_to_chunk[future]
        try:
            chunk_results = future.result()
            # Only update with new results
            for idx, result in chunk_results.items():
                if idx not in processed_indices:
                    results_dict[idx] = result
                    processed_indices.add(idx)
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")

# Update the dataframe with results
result_columns = ["translated_text", "language_code", "predicted_label", "confidence_score"]
for idx, result_tuple in results_dict.items():
    valid_data.loc[idx, result_columns] = result_tuple

# Save results
output_path = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/pitch26/CSAT/sp-en-transit.csv"
valid_data.to_csv(output_path, index=False, storage_options={"sas_token": base64.b64decode(sas_token).decode()})
print("Output saved to: ", output_path)
print(f"Total processed records: {len(results_dict)}")
print(f"Total input records: {len(valid_comments)}")

