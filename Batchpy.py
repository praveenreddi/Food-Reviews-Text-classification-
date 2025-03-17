def get_system_message(formatted_comments):
    system_message = f"""
    Analyze the following customer comments. Create a concise summary of the key issues
    users are experiencing. Format the summary as paragraphs that start with "Customers are facing..."
    and use clear, direct language.

    Comments to analyze:
    {formatted_comments}
    """
    return system_message

def get_final_summary_message(batch_summaries):
    system_message = f"""
    Combine these summaries into one concise, comprehensive summary of customer issues.
    Format the final summary as paragraphs that start with "Customers are facing..." 
    and use clear, direct language.

    Summaries to combine:
    {batch_summaries}
    """
    return system_message

def process_comments_in_two_batches(comments, char_limit=3000):
    # Split all comments into exactly two batches
    mid_point = len(comments) // 2
    batch1 = comments[:mid_point]
    batch2 = comments[mid_point:]
    
    all_responses = []
    
    # Process each batch
    for i, batch in enumerate([batch1, batch2]):
        batch_num = i + 1
        formatted_batch = "\n".join([f"- {comment}" for comment in batch])
        
        # Truncate if too long
        if len(formatted_batch) > char_limit:
            formatted_batch = formatted_batch[:char_limit] + "... [truncated]"
        
        print(f"Processing batch {batch_num}/2, length: {len(formatted_batch)} chars")
        
        system_message = get_system_message(formatted_batch)
        messages = [{'role': 'system', 'content': system_message}]
        
        try:
            print(f"Calling LLM for batch {batch_num}...")
            completion = llm_call(messages)
            
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                response = completion.choices[0].message.content
                response = response.strip('\"\'')
                all_responses.append(response)
                print(f"Successfully processed batch {batch_num}")
            else:
                error_msg = f"Failed to process batch {batch_num}: LLM returned invalid response"
                all_responses.append(error_msg)
                print(error_msg)
        except Exception as e:
            error_msg = f"Error in batch {batch_num}: {str(e)}"
            all_responses.append(error_msg)
            print(error_msg)
    
    # Combine the two batch summaries into one final summary
    if len(all_responses) == 2:
        batch_summaries = "\n\n".join(all_responses)
        
        final_summary_message = get_final_summary_message(batch_summaries)
        final_messages = [{'role': 'system', 'content': final_summary_message}]
        
        try:
            print("Creating final combined summary...")
            final_completion = llm_call(final_messages)
            
            if final_completion is not None and hasattr(final_completion, 'choices') and len(final_completion.choices) > 0:
                final_response = final_completion.choices[0].message.content
                final_response = final_response.strip('\"\'')
                print("Successfully created final summary")
                return final_response
            else:
                return "Failed to create final summary: LLM returned invalid response"
        except Exception as e:
            return f"Error creating final summary: {str(e)}"
    else:
        return "\n\n".join(all_responses)

# Main execution
try:
    # Get all comments from dataframe
    all_comments = df['Comments'].tolist()
    
    # Remove any None values or empty strings
    all_comments = [comment for comment in all_comments if comment and isinstance(comment, str)]
    
    print(f"Total comments: {len(all_comments)}")
    
    # Process comments in exactly two batches and create final summary
    final_summary = process_comments_in_two_batches(all_comments, char_limit=3000)
    
    # Print the final summarized response
    print("\n\nFINAL SUMMARY OF CUSTOMER ISSUES:")
    print("==================================")
    print(final_summary)
    
except Exception as e:
    print(f"An error occurred in the main execution: {str(e)}")
