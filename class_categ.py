def classify_comment_rationale(user_comment):
    system_message = get_system_message()
    messages = get_prompt_message(system_message, user_comment)
    try:
        response_text = llm_call(messages)
        if not response_text:
            return ("Error", "Error", "Error", 0)

        # Clean and standardize the response text
        try:
            # Remove any whitespace and newlines
            response_text = response_text.strip()

            # Find the JSON structure
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > 0:
                response_text = response_text[start_idx:end_idx]

                # Clean up common JSON formatting issues
                response_text = response_text.replace("'", '"')  # Replace single quotes
                response_text = response_text.replace('\n', '')  # Remove newlines
                response_text = response_text.replace('\\', '')  # Remove escape characters

                # Fix missing commas between properties (if any)
                response_text = response_text.replace('} {', '}, {')

                # Try to parse the cleaned JSON
                response_json = json.loads(response_text)
            else:
                # If no JSON structure found, try ast.literal_eval
                response_dict = ast.literal_eval(response_text)
                response_json = json.loads(json.dumps(response_dict))

        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Problematic response: {response_text}")
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
