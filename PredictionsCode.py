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

def classify_comment_rationale(user_comment):
    system_message = get_system_message_modified()
    messages = build_prompt_message_modified(system_message, user_comment)

    try:
        completion = llm_call(messages)

        print(completion)

        try:
            response_json = json.loads(completion)
            print(f"\nParsed JSON: {response_json}")

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
