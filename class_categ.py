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

                # Enhanced cleaning
                response_text = response_text.replace("'", '"')
                response_text = response_text.replace('\n', '')
                response_text = response_text.replace('""', '"')

                # Fix common JSON formatting issues
                response_text = response_text.replace('}{', '},{')
                response_text = response_text.replace('" "', '", "')
                response_text = response_text.replace('"EN_text"', '"EN_text":')
                response_text = response_text.replace('"language_code"', '"language_code":')
                response_text = response_text.replace('"predicted_label"', '"predicted_label":')
                response_text = response_text.replace('"confidence_score"', '"confidence_score":')

                print(f"Cleaned response: {response_text}")  # Debug print

                # Try multiple parsing approaches
                try:
                    response_json = json.loads(response_text)
                except json.JSONDecodeError:
                    try:
                        # Add missing commas between key-value pairs
                        response_text = re.sub(r'(["}])(["{])', r'\1,\2', response_text)
                        response_json = json.loads(response_text)
                    except:
                        # Last resort: ast.literal_eval
                        response_dict = ast.literal_eval(response_text)
                        if isinstance(response_dict, dict):
                            response_json = response_dict
                        else:
                            print(f"Failed to parse as dictionary: {response_dict}")
                            return ("Error", "Error", "Error", 0)
            else:
                print("No JSON structure found")
                return ("Error", "Error", "Error", 0)

        except Exception as e:
            print(f"Parsing error: {str(e)}")
            print(f"Problematic response: {response_text}")
            return ("Error", "Error", "Error", 0)

        # Ensure all required fields are present
        return (
            str(response_json.get("EN_text", "")),
            str(response_json.get("language_code", "")),
            str(response_json.get("predicted_label", "unknown/vague")),
            float(response_json.get("confidence_score", 0.5))
        )
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return ("Error", "Error", "Error", 0)
