def classify_comment_rationale(user_comment):
    try:
        system_message = get_system_message()
        messages = get_prompt_message(system_message, user_comment)

        response_text = llm._call(messages)

        if not response_text:
            print(f"\nText causing empty response error:\n{user_comment}\n")
            return ("Error", "Error", "Error", 0)

        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                print(f"\nText causing JSON parsing error:\n{user_comment}\n")
                print(f"Response received: {response_text}\n")
                return ("Error", "Error", "Error", 0)

            json_str = response_text[start_idx:end_idx]
            response_json = json.loads(json_str)

            return (
                response_json.get("EN_text", ""),
                response_json.get("language_code", ""),
                response_json.get("predicted_label", "unknown/vague"),
                float(response_json.get("confidence_score", 0.5))
            )

        except (json.JSONDecodeError, ValueError) as e:
            print(f"\nText causing JSON decode error:\n{user_comment}\n")
            print(f"Error: {str(e)}\n")
            return ("Error", "Error", "Error", 0)

    except Exception as e:
        print(f"\nText causing general error:\n{user_comment}\n")
        print(f"Error: {str(e)}\n")
        return ("Error", "Error", "Error", 0)
