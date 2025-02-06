system_message = """Analyze the sentiment and emotion in the following comment.
    Classify the emotion into one of these specific categories:
    Happy, Sad, Angry, Frustration, Sarcastic, Neutral, Surprised, Fear, Unhappy, or Unknown.

    Respond with a JSON object containing:
    {
        "emotion_summary": one of the specified emotions listed above,
        "emotion_confidence": confidence score between 0 and 1
    }

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
    - Always respond with exactly one of these 10 emotions: Happy, Sad, Angry, Frustration, Sarcastic, Neutral, Surprised, Fear, Unhappy, or Unknown
    - Use 'Unknown' when you're not confident about the emotion or when the text is unclear
    - If confidence is below 0.4, default to 'Unknown'
    - Do not use any other emotion categories"""
