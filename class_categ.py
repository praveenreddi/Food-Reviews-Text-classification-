def classify_comment_rationale(user_comment):
    system_message = get_system_message()
    messages = get_prompt_message(system_message, user_comment)
    try:
        response_text = llm_call(messages)
        if not response_text:
            return ("Error", "Error", "Error", 0)

        print(f"Original response: {response_text}")  # Debug print

        try:
            # Clean the response text
            response_text = response_text.strip()

            # Extract the dictionary part
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > 0:
                response_text = response_text[start_idx:end_idx]
                # Basic cleaning
                response_text = response_text.replace("'", '"')
                response_text = response_text.replace('\n', '')

                print(f"Cleaned response: {response_text}")  # Debug print

                # Try parsing as JSON first
                try:
                    response_json = json.loads(response_text)
                except:
                    # If JSON fails, try ast.literal_eval
                    response_dict = ast.literal_eval(response_text)
                    if isinstance(response_dict, dict):
                        response_json = response_dict
                    else:
                        return ("Error", "Error", "Error", 0)
            else:
                return ("Error", "Error", "Error", 0)

        except Exception as e:
            print(f"Parsing error: {str(e)}")
            print(f"Problematic response: {response_text}")
            return ("Error", "Error", "Error", 0)

        return (
            str(response_json.get("EN_text", "")),
            str(response_json.get("language_code", "")),
            str(response_json.get("predicted_label", "unknown/vague")),
            float(response_json.get("confidence_score", 0.5))
        )
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return ("Error", "Error", "Error", 0)








if start_idx != -1 and end_idx > 0:
    response_text = response_text[start_idx:end_idx]
    # Enhanced cleaning for string literals
    response_text = response_text.replace('\\"', '"')     # Fix escaped quotes
    response_text = response_text.replace("'", '"')       # Replace single quotes
    response_text = response_text.replace('\n', '')       # Remove newlines
    response_text = response_text.replace('"""', '"')     # Fix triple quotes
    response_text = response_text.replace('""', '"')      # Fix double quotes

    # Ensure quotes are properly terminated
    if response_text.count('"') % 2 != 0:
        response_text = response_text.replace('"', "'")   # Fall back to single quotes if unmatched

    print(f"Cleaned response: {response_text}")  # Debug print
