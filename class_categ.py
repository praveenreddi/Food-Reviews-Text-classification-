def classify_comment_rationale(user_comment):
    system_message = get_system_message()
    messages = get_prompt_message(system_message, user_comment)
    try:
        response_text = llm_call(messages)
        if not response_text:
            return ("Error", "Error", "Error", 0)
        
        # Debug print to see raw response
        print(f"Raw response: {response_text}")
        
        try:
            # First clean the response text
            response_text = response_text.strip()
            
            # If response is wrapped in quotes, remove them
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text[1:-1]
            
            # Find the dictionary-like structure
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > 0:
                response_text = response_text[start_idx:end_idx]
                
                # Replace problematic characters
                response_text = (response_text
                    .replace("'", '"')
                    .replace('\n', '')
                    .replace('\\', '')
                    .replace('} {', '}, {')
                    .replace('""', '"')
                )
                
                # Fix common JSON formatting issues
                response_text = re.sub(r'(["}])([A-Za-z])', r'\1, \2', response_text)
                response_text = re.sub(r'([^,{])\s*"', r'\1, "', response_text)
                
                print(f"Cleaned response: {response_text}")  # Debug print
                
                try:
                    response_json = json.loads(response_text)
                except:
                    # If JSON parsing fails, try ast.literal_eval
                    response_dict = ast.literal_eval(response_text)
                    response_json = json.loads(json.dumps(response_dict))
            else:
                print("No JSON structure found in response")
                return ("Error", "Error", "Error", 0)
                
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Problematic response: {response_text}")
            
            # Last resort: try to extract values using regex
            try:
                en_text = re.search(r'"EN_text"\s*:\s*"([^"]*)"', response_text)
                lang_code = re.search(r'"language_code"\s*:\s*"([^"]*)"', response_text)
                pred_label = re.search(r'"predicted_label"\s*:\s*"([^"]*)"', response_text)
                conf_score = re.search(r'"confidence_score"\s*:\s*([\d.]+)', response_text)
                
                return (
                    en_text.group(1) if en_text else "Error",
                    lang_code.group(1) if lang_code else "Error",
                    pred_label.group(1) if pred_label else "Error",
                    float(conf_score.group(1)) if conf_score else 0
                )
            except:
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
