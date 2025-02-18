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

            # Remove any text before the first { and after the last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                print("No valid JSON structure found")
                return ("Error", "Error", "Error", 0)

            response_text = response_text[start_idx:end_idx]

            # Basic string replacements
            response_text = (response_text
                .replace("'", '"')        # Replace single quotes with double quotes
                .replace('\n', '')        # Remove newlines
                .replace('\r', '')        # Remove carriage returns
                .replace('\\', '')        # Remove backslashes
            )

            print(f"After basic cleaning: {response_text}")  # Debug print

            # Try to parse the response
            try:
                # First attempt: direct JSON parsing
                response_json = json.loads(response_text)
            except:
                try:
                    # Second attempt: convert string to dictionary using ast
                    print("Attempting ast.literal_eval...")
                    response_dict = ast.literal_eval(response_text)
                    print(f"ast.literal_eval result: {response_dict}")

                    if isinstance(response_dict, dict):
                        response_json = response_dict
                    else:
                        print(f"Unexpected type after ast.literal_eval: {type(response_dict)}")
                        return ("Error", "Error", "Error", 0)
                except Exception as ast_error:
                    print(f"ast.literal_eval failed: {str(ast_error)}")

                    # Create a basic response if parsing fails
                    print("Creating basic response...")
                    response_json = {
                        "EN_text": "",
                        "language_code": "",
                        "predicted_label": "unknown/vague",
                        "confidence_score": 0.5
                    }

        except Exception as e:
            print(f"Parsing error details: {str(e)}")
            print(f"Response type: {type(response_text)}")
            print(f"Problematic response: {response_text}")
            return ("Error", "Error", "Error", 0)

        # Extract values with default fallbacks
        return (
            str(response_json.get("EN_text", "")),
            str(response_json.get("language_code", "")),
            str(response_json.get("predicted_label", "unknown/vague")),
            float(response_json.get("confidence_score", 0.5))
        )

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        print(f"Full error details: {e.__class__.__name__}: {str(e)}")
        return ("Error", "Error", "Error", 0)
