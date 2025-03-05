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


