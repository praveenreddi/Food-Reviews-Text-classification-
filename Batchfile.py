Here's the complete error handling code for your LLM call:

```python
# Define your get_system_message function
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

# Get all comments
all_comments = df['Comments'].tolist()

# Format all comments
formatted_comments = "\n".join([f"- {comment}" for comment in all_comments])

# Create system message
system_message = get_system_message(formatted_comments)

# Create messages for LLM
messages = [
    {'role': 'system', 'content': system_message}
]

# Call LLM with error handling
try:
    completion = llm_call(messages)
    if completion and hasattr(completion, 'choices') and len(completion.choices) > 0:
        response = completion.choices[0].message.content
        response = response.strip('\"\'')
    else:
        response = "Failed to get response from LLM service"
except Exception as e:
    response = f"Error occurred: {str(e)}"

# Print or use the response
print(response)
```

If you're still having token limit issues, you can implement batching:

```python
# For handling large comment sets with batching
def process_comments_in_batches(comments, batch_size=50):
    all_responses = []
    
    # Process comments in batches
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        formatted_batch = "\n".join([f"- {comment}" for comment in batch])
        
        system_message = get_system_message(formatted_batch)
        messages = [{'role': 'system', 'content': system_message}]
        
        try:
            completion = llm_call(messages)
            if completion and hasattr(completion, 'choices') and len(completion.choices) > 0:
                response = completion.choices[0].message.content
                response = response.strip('\"\'')
                all_responses.append(response)
            else:
                all_responses.append(f"Failed to process batch {i//batch_size + 1}")
        except Exception as e:
            all_responses.append(f"Error in batch {i//batch_size + 1}: {str(e)}")
    
    # Combine all batch responses
    return "\n\n".join(all_responses)

# Use the batching function
final_response = process_comments_in_batches(all_comments, batch_size=50)
print(final_response)
```
