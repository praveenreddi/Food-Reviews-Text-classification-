def classify_comment_rationale(user_comment):
    system_message = get_system_message()
    messages = get_prompt_message(system_message, user_comment)
    try:
        response_text = llm_call(messages)
        if not response_text:
            return ("Error", "Error", "Error", 0)

        # Clean up the response text to ensure valid JSON
        try:
            # First attempt: direct JSON parsing
            response_json = json.loads(response_text)
        except:
            try:
                # Second attempt: using ast to evaluate the dictionary string
                response_text = response_text.strip()
                if response_text.startswith("'") and response_text.endswith("'"):
                    response_text = response_text[1:-1]
                response_dict = ast.literal_eval(response_text)
                response_json = json.loads(json.dumps(response_dict))
            except:
                # Third attempt: find and extract JSON-like structure
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > 0:
                    response_text = response_text[start_idx:end_idx]
                    response_text = response_text.replace("'", '"')  # Replace single quotes with double quotes
                    response_json = json.loads(response_text)
                else:
                    return ("Error", "Error", "Error", 0)

        return (
            response_json.get("EN_text", ""),
            response_json.get("language_code", ""),
            response_json.get("predicted_label", "unknown/vague"),
            float(response_json.get("confidence_score", 0.5))
        )
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return ("Error", "Error", "Error", 0)
