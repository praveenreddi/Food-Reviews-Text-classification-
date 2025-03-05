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
