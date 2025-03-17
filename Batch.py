def get_system_message(formatted_comments):
    system_message = f"""
    Analyze the following customer comments. Create a concise summary of the key issues
    grouped under appropriate headings based on common themes you identify in the comments.

    For each theme, create a section with a clear heading and summarize the specific problems
    users are experiencing related to that theme. Format each theme summary as a paragraph
    that MUST start with "Customers are facing..." and use clear, direct language.

    Prioritize issues by frequency and severity. Include specific examples where relevant.

    Comments to analyze:
    {formatted_comments}
    """
    return system_message

def process_comments_in_batches(comments, batch_size=10, char_limit=3000):
    all_responses = []
    
    # If we have too many comments, sample a subset
    if len(comments) > 100:
        import random
        # Set seed for reproducibility
        random.seed(42)
        sampled_comments = random.sample(comments, 100)
        print(f"Sampling 100 comments from {len(comments)} total comments")
    else:
        sampled_comments = comments
    
    for i in range(0, len(sampled_comments), batch_size):
        batch = sampled_comments[i:i+batch_size]
        formatted_batch = "\n".join([f"- {comment}" for comment in batch])
        
        # Truncate if too long
        if len(formatted_batch) > char_limit:
            formatted_batch = formatted_batch[:char_limit] + "... [truncated]"
        
        print(f"Processing batch {i//batch_size + 1}/{(len(sampled_comments)-1)//batch_size + 1}, length: {len(formatted_batch)} chars")
        
        system_message = get_system_message(formatted_batch)
        messages = [{'role': 'system', 'content': system_message}]
        
        try:
            print(f"Calling LLM for batch {i//batch_size + 1}...")
            completion = llm_call(messages)
            print(f"Response received for batch {i//batch_size + 1}: {type(completion)}")
            
            # Check if completion exists and has the expected structure
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                response = completion.choices[0].message.content
                response = response.strip('\"\'')
                all_responses.append(response)
                print(f"Successfully processed batch {i//batch_size + 1}")
            else:
                error_msg = f"Failed to process batch {i//batch_size + 1}: LLM returned invalid response"
                all_responses.append(error_msg)
                print(error_msg)
        except AttributeError as e:
            error_msg = f"Error in batch {i//batch_size + 1}: LLM response format error - {str(e)}"
            all_responses.append(error_msg)
            print(error_msg)
        except Exception as e:
            error_msg = f"Error in batch {i//batch_size + 1}: {str(e)}"
            all_responses.append(error_msg)
            print(error_msg)
        
        # Add a small delay between API calls to avoid rate limiting
        import time
        time.sleep(1)
    
    return "\n\n".join(all_responses)

# Main execution
try:
    # Get all comments from dataframe
    all_comments = df['Comments'].tolist()
    
    # Remove any None values or empty strings
    all_comments = [comment for comment in all_comments if comment and isinstance(comment, str)]
    
    print(f"Total comments: {len(all_comments)}")
    print(f"Total characters in all comments: {sum(len(c) for c in all_comments)}")
    
    # Process comments in small batches with character limits
    final_response = process_comments_in_batches(all_comments, batch_size=5, char_limit=2000)
    
    # Print the final summarized response
    print("\n\nFINAL SUMMARY OF CUSTOMER ISSUES:")
    print("==================================")
    print(final_response)
    
except Exception as e:
    print(f"An error occurred in the main execution: {str(e)}")
