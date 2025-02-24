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



def classify_single_comment(text, idx):
    """Process a single comment and handle errors individually"""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return idx, ("Error - Empty Text", "Error", "Error", 0)
    
    try:
        # Define your GPT prompt
        system_message = """
        You have two tasks:
        1. Translate the given text to English if it's not in English
        2. Classify the text into appropriate category
        
        Respond in the following JSON format only:
        {
            "translation": {
                "english_text": "translated text here",
                "source_language": "detected language code"
            },
            "classification": {
                "category": "category name",
                "confidence": confidence_score
            }
        }
        """
        
        # Prepare messages for GPT
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": str(text)}
        ]
        
        # Call GPT API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or your preferred GPT model
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            # Extract response
            response_text = response['choices'][0]['message']['content']
            
            # Clean and parse response
            response_text = response_text.strip()
            
            # Find the JSON part in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > 0:
                response_text = response_text[start_idx:end_idx]
                
                try:
                    # Parse JSON response
                    response_json = json.loads(response_text)
                    
                    # Extract required fields from GPT response
                    english_text = response_json.get('translation', {}).get('english_text', '')
                    language_code = response_json.get('translation', {}).get('source_language', '')
                    category = response_json.get('classification', {}).get('category', 'unknown')
                    confidence = float(response_json.get('classification', {}).get('confidence', 0.5))
                    
                    return idx, (
                        str(english_text),
                        str(language_code),
                        str(category),
                        confidence
                    )
                    
                except json.JSONDecodeError:
                    # Try parsing with ast if JSON parsing fails
                    try:
                        response_dict = ast.literal_eval(response_text)
                        if isinstance(response_dict, dict):
                            english_text = response_dict.get('translation', {}).get('english_text', '')
                            language_code = response_dict.get('translation', {}).get('source_language', '')
                            category = response_dict.get('classification', {}).get('category', 'unknown')
                            confidence = float(response_dict.get('classification', {}).get('confidence', 0.5))
                            
                            return idx, (
                                str(english_text),
                                str(language_code),
                                str(category),
                                confidence
                            )
                    except:
                        return idx, ("Error - Parse Failed", "Error", "Error", 0)
            
            return idx, ("Error - Invalid JSON", "Error", "Error", 0)
            
        except Exception as api_error:
            return idx, (f"Error - API Call Failed: {str(api_error)}", "Error", "Error", 0)
            
    except Exception as e:
        return idx, (f"Error - General: {str(e)}", "Error", "Error", 0)

# Add OpenAI API configuration at the start of your code
import openai
openai.api_key = 'your-api-key'  # Replace with your API key

# The rest of your code remains the same
