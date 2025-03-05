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


