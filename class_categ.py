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
    # First clean newlines and backslashes
    response_text = response_text.replace('\n', ' ')
    response_text = response_text.replace('\\', '')

    # Split into key-value pairs using a more lenient pattern
    pattern = r'"([^"]+)"\s*:\s*"([^"}]*)'
    matches = re.findall(pattern, response_text)

    # Rebuild the JSON properly
    cleaned_json = "{"
    for i, (key, value) in enumerate(matches):
        # Clean the key and value
        key = key.strip()
        value = value.strip()

        # Add the cleaned key-value pair
        cleaned_json += f'"{key}":"{value}"'

        # Add comma if not the last pair
        if i < len(matches) - 1:
            cleaned_json += ","

    cleaned_json += "}"
    response_text = cleaned_json

    print(f"Cleaned response: {response_text}")  # Debug print
