def get_system_message_modified():
    # Define the delimiter as a variable
    delimiter = "####"
    
    # Define the emotion labels as a list
    emotion_labels = [
        "Happy", 
        "Sad", 
        "Angry", 
        "Frustration", 
        "Sarcastic", 
        "Neutral", 
        "Surprised", 
        "Fear", 
        "Unhappy", 
        "Unknown"
    ]
    
    # Convert the list to a comma-separated string for the prompt
    labels_str = ", ".join(emotion_labels)
    
    # Create the system message with the variables
    system_message = f"""You are an advanced AI assistant analyzing user comments and user comments will be delimited with {delimiter} characters.
Classify the emotion into one of these specific categories:
{labels_str}.

Return with a JSON object containing:
{{
    "emotion_summary": one of the specified emotions listed above,
    "emotion_confidence": confidence score between 0 and 1
}}
""".strip()
    
    return system_message



def get_system_message_modified():
    delimiter = "####"
    emotion_labels = [
        "Happy", "Sad", "Angry", "Frustration",
        "Sarcastic", "Neutral", "Surprised",
        "Fear", "Unhappy", "Unknown"
    ]
    labels_str = ", ".join(emotion_labels)

    system_message = f"""You are an advanced AI assistant analyzing user comments and user comments will be delimited with {delimiter} characters.
Classify the emotion into one of these specific categories:
{labels_str}.

Return with a JSON object containing:
{{
    "emotion_summary": one of the specified emotions listed above,
    "emotion_confidence": confidence score between 0 and 1
}}

IMPORTANT: Do not include the delimiter '{delimiter}' in your response. Return only the JSON object.
""".strip()

    return system_message


try:
    # Print the raw completion for debugging
    print(f"Raw completion: {completion}")

    # Clean the response by removing the delimiter if present
    cleaned_completion = completion
    if delimiter in completion:
        # Extract just the JSON part between or after delimiters
        parts = completion.split(delimiter)
        for part in parts:
            if "{" in part and "}" in part:
                # Find the JSON object in the part
                start = part.find("{")
                end = part.rfind("}") + 1
                if start >= 0 and end > start:
                    cleaned_completion = part[start:end]
                    break

    # Now try to parse the cleaned JSON
    response_json = json.loads(cleaned_completion)
    print(f"Parsed JSON: {response_json}")

    # Rest of your code remains the same
    emotion = response_json.get("emotion_summary", "neutral")
    confidence = float(response_json.get("emotion_confidence", 0.5))

    # ... existing code ...


labels = [
    "unable_to_view_bill",
    "overcharged",
    "billing_history_related",
    "unable_to_download_bill",
    "billing_error",
    "virtual_agent_related",
    "chat_disappear/disconnect_issues",
    "chat_agent_related",
    "chat_not_responding",
    "customer_service_agent_related",
    "unable_to_login",
    "credentials_not_accepted",
    "unable_to_logout_find_logout_option",
    "code_related",
    "change_plan_related",
    "cancel_service_line_account",
    "unable_to_access_account_info",
    "manage_wireless_services",
    "international_plan_related",
    "newsmax_related",
    "manage_device_related",
    "account_sync_issues",
    "hbo_related",
    "unable_to_update_account_info",
    "navigation_issues",
    "unable_to_make_payment",
    "autopay_related",
    "unable_to_update_payment_method",
    "payment_arrangement",
    "payment_confirmation_related",
    "slow_webpage/page_load_issues",
    "redirection_issue",
    "webpage_error/crash/break",
    "long_wait_times_to_connect_with_agent",
    "unable_to_update_profile_info",
    "unable_to_register",
    "unable_to_link_accounts",
    "registration_generic_error",
    "unable_to_view_usage",
    "missing_usage_logs",
    "logs_not_updated",
    "error_message",
    "phone_network_service",
    "order_status",
    "order_cancel"
]





import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Load your Excel data
df = pd.read_excel("Cancelled CDEX.xlsx")

# 2. Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create document list for vector store
documents = []
for idx, row in df.iterrows():
    content = f"Summary: {row['Summary']}\nIssue key: {row['Key']}"
    documents.append({"page_content": content, "metadata": {"source": "customer_complaints"}})

# Create vector store
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Define classification function
def classify_summary(summary):
    # Prepare query
    query = {"Summary": summary}

    # Retrieve similar documents
    docs = retriever.get_relevant_documents(str(query))

    # Extract summaries from retrieved documents
    summaries = []
    for doc in docs:
        if "Summary:" in doc.page_content:
            summary_text = doc.page_content.split("Summary:")[1].split("\n")[0].strip()
            summaries.append(summary_text)

    # Return most frequent summary as the predicted label
    if summaries:
        return max(set(summaries), key=summaries.count)
    else:
        return "unknown"

# 4. Process all summaries from the dataset
all_summaries = df['Summary'].tolist()
results = []

for summary in all_summaries:
    predicted_label = classify_summary(summary)
    results.append({
        "original_summary": summary,
        "predicted_label": predicted_label
    })

# 5. Create results DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# 6. Export to Excel
results_df.to_excel("all_summaries_classification.xlsx", index=False)





import pandas as pd

# Load your Excel file
df = pd.read_excel('your_excel_file.xlsx')

# Filter for profile_related and registration_related comments
profile_comments = df[df['prediction_theme_label'] == 'profile_related']['comments'].tolist()
registration_comments = df[df['prediction_theme_label'] == 'registration_related']['comments'].tolist()

# Combine the relevant comments
relevant_comments = profile_comments + registration_comments

# Format the comments for the prompt
formatted_comments = "\n".join([f"- {comment}" for comment in relevant_comments])

# Create the prompt for Llama 3
llama_prompt = f"""
Analyze the following customer comments that have been labeled as profile_related or registration_related. Create a concise summary of the key issues under the heading "Registration & Login". Focus on specific problems users are experiencing, error messages, redirects, and process failures. Format the summary as a paragraph with clear, direct language.

Comments to analyze:
{formatted_comments}
"""

print(llama_prompt)
# You would then send this prompt to your Llama 3 API


def get_system_message(formatted_comments):
    system_message = f"""
Analyze the following customer comments that have been labeled as profile_related or registration_related. Create a concise summary of the key issues under
the heading "Registration & Login". Focus on specific problems users are experiencing, error messages, redirects, and process failures. Format the summary as
a paragraph that MUST start with "Customers are facing..." and use clear, direct language.

Comments to analyze:
{formatted_comments}
"""
    return system_message

# Filter for profile_related and registration_related comments
profile_comments = df[df['prediction_theme_label'] == 'profile_related']['comments'].tolist()
registration_comments = df[df['prediction_theme_label'] == 'registration_related']['comments'].tolist()

# Combine the relevant comments
relevant_comments = profile_comments + registration_comments

# Format the comments for the prompt
formatted_comments = "\n".join([f"- {comment}" for comment in relevant_comments])

# Get the system message with formatted comments
system_message = get_system_message(formatted_comments)

# Prepare messages for LLM
messages = [
    {'role': 'system', 'content': system_message}
]

# Call the LLM
completion = llm.call(messages)
response = completion.choices[0].message.content

# If the response is in JSON format, extract the predicted label
try:
    response = json.loads(response)['predicted_label']
except:
    # If not in JSON format, use the response as is
    pass

print(response)

def get_system_message(formatted_comments):
    system_message = f"""
Analyze the following customer comments that have been labeled as profile_related or registration_related. Create a concise summary of the key issues under
the heading "Registration & Login". Focus on specific problems users are experiencing, error messages, redirects, and process failures. Format the summary as
a paragraph that MUST start with "Customers are facing..." and use clear, direct language.

Comments to analyze:
{formatted_comments}

IMPORTANT: Your summary must ONLY include information related to registration, login, profile management, and account access. Do NOT include any information about payments, orders, shipping, product quality, or other unrelated topics.
"""
    return system_message

# Filter for profile_related and registration_related comments
profile_comments = df[df['prediction_theme_label'] == 'profile_related']['comments'].tolist()
registration_comments = df[df['prediction_theme_label'] == 'registration_related']['comments'].tolist()

# Combine the relevant comments
relevant_comments = profile_comments + registration_comments

# Format the comments for the prompt
formatted_comments = "\n".join([f"- {comment}" for comment in relevant_comments])

# Get the system message with formatted comments
system_message = get_system_message(formatted_comments)

# Prepare messages for LLM
messages = [
    {'role': 'system', 'content': system_message}
]

# Call the LLM
completion = llm.call(messages)
response = completion.choices[0].message.content

# If the response is in JSON format, extract the predicted label
try:
    response = json.loads(response)['predicted_label']
except:
    # If not in JSON format, use the response as is
    pass

# Verification step to ensure only registration and login related content
def verify_response(response):
    verification_prompt = f"""
    Review this summary and verify it ONLY contains information related to registration, login, profile management, and account access issues.
    Remove any content about payments, orders, shipping, product quality, or other unrelated topics.
    Ensure the summary starts with "Customers are facing..." and maintains a clear, concise format.
    
    Summary to verify:
    {response}
    
    Return ONLY the verified and corrected summary.
    """
    
    verification_messages = [
        {'role': 'system', 'content': verification_prompt}
    ]
    
    verification_completion = llm.call(verification_messages)
    verified_response = verification_completion.choices[0].message.content
    
    # Clean up any formatting or quotes that might be added
    verified_response = verified_response.strip('"\'')
    
    return verified_response

# Apply verification
final_response = verify_response(response)
print(final_response)
