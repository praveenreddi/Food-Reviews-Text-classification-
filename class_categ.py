def analyze_feedback_by_category(df):
    llm = LlamaLLM()

    analysis_prompt = """Analyze all customer feedback and provide a detailed summary following this exact format:

    Web Acct Mgmt - Below are the themes observed for Web Acct Mgmt journey this week are -

    For each category, list ONLY the categories with significant issues, and format as follows:
    [Category Name] - [List main issues with their QM links in brackets]

    Example format:
    Registration & Login - Customers are unable to reset password [QM link1, QM link2], Users cannot complete registration process [QM link3, QM link4]

    Important rules:
    1. Group similar issues together
    2. Include QM links in brackets after each issue
    3. Use bullet points for separate categories
    4. Only include categories with significant issues
    5. For each issue, include 2-4 relevant QM links
    6. Format QM links as: (https://att.quantummetric.com/#/[ID])

    Analyze these comments and their QM links:
    {comments_and_links}
    """

    # Group comments and QM links by category
    grouped_data = df.groupby('Category').agg({
        'Comments': list,
        'att18_@MReplayLink': lambda x: list(x.dropna())
    }).reset_index()

    # Prepare input for analysis
    analysis_input = ""
    for _, row in grouped_data.iterrows():
        analysis_input += f"\nCategory: {row['Category']}\n"
        for comment, link in zip(row['Comments'], row['att18_@MReplayLink']):
            analysis_input += f"Comment: {comment}\nQM Link: {link}\n"

    # Get analysis from LLM
    try:
        result = llm(analysis_prompt.format(comments_and_links=analysis_input))
        return result
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None

# Use the function
try:
    # Get the analysis
    analysis = analyze_feedback_by_category(df)

    # Print the results
    print(analysis)

    # Optionally save to file
    with open('feedback_analysis.txt', 'w') as f:
        f.write(analysis)

except Exception as e:
    print(f"Error: {e}")




----------------------------

def format_analysis(analysis_text):
    formatted_prompt = """Take this analysis and format it exactly like this example:

    Web Acct Mgmt - Below are the themes observed for Web Acct Mgmt journey this week are -
    Registration & Login - [Issues with QM links]
    Profile - [Issues with QM links]
    ViewBill & Payment - [Issues with QM links]
    TOBR - [Issues with QM links]
    View Usage - [Issues with QM links]
    Web Sales - [Issues with QM links]
    Upper Funnel - [Issues with QM links]
    Support & Chat - [Issues with QM links]
    Device Unlock - [Issues with QM links]
    Outage Portal - [Issues with QM links]

    Format rules:
    1. Start with "Web Acct Mgmt - Below are the themes observed..."
    2. List each category with a bullet point
    3. Include QM links in parentheses
    4. Group similar issues together
    5. Separate different issues within same category with commas

    Original analysis:
    {analysis}
    """

    llm = LlamaLLM()
    try:
        formatted_result = llm(formatted_prompt.format(analysis=analysis_text))
        return formatted_result
    except Exception as e:
        print(f"Error in formatting: {e}")
        return analysis_text

# Use both functions
try:
    # Get initial analysis
    initial_analysis = analyze_feedback_by_category(df)

    # Format the analysis
    final_analysis = format_analysis(initial_analysis)

    # Print results
    print(final_analysis)

    # Save to file
    with open('feedback_analysis.txt', 'w') as f:
        f.write(final_analysis)

except Exception as e:
    print(f"Error: {e}")



-------------------------------------

def analyze_feedback_by_category(df):
    llm = LlamaLLM()

    analysis_prompt = """Analyze all customer feedback and provide a comprehensive summary in exactly this format:

    Web Acct Mgmt - Below are the themes observed for Web Acct Mgmt journey this week are -

    Registration & Login - [List main issues with their exact QM links in parentheses]

    Profile - [List main issues with their exact QM links in parentheses]

    ViewBill & Payment - [List main issues with their exact QM links in parentheses]

    TOBR - [List main issues with their exact QM links in parentheses]

    View Usage - [List main issues with their exact QM links in parentheses]

    Web Sales - [List main issues with their exact QM links in parentheses]

    Upper Funnel - [List main issues with their exact QM links in parentheses]

    Support & Chat - Support & Chat journey weekly avg rating has improved this week and above the last month avg rating. Some common feedback observed in customer feedback on support & chat are as below.
    • Online Support - [List main issues related to chat, hold times, and customer service]
    • Device Unlock - [List main issues with their exact QM links in parentheses]
    • Outage Portal - [List main issues with their exact QM links in parentheses]

    Rules:
    1. Exclude the "Unknown" category
    2. For each issue, include the QM links in parentheses
    3. Group similar issues together
    4. Format QM links exactly as: (https://att.quantummetric.com/#/[ID])
    5. Under Support & Chat, group issues into Online Support, Device Unlock, and Outage Portal
    6. For Online Support, include customer feedback about wait times, chat issues, and service quality
    7. Separate different issues within the same category using commas

    Analyze these comments and their QM links:
    {comments_and_links}
    """

    # Filter out Unknown category and group data
    df_filtered = df[df['Category'] != 'Unknown']
    grouped_data = df_filtered.groupby('Category').agg({
        'Comments': list,
        'att18_@MReplayLink': lambda x: list(x.dropna())
    }).reset_index()

    # Prepare input for analysis
    analysis_input = ""
    for _, row in grouped_data.iterrows():
        analysis_input += f"\nCategory: {row['Category']}\n"
        for comment, link in zip(row['Comments'], row['att18_@MReplayLink']):
            analysis_input += f"Comment: {comment}\nQM Link: {link}\n"

    try:
        result = llm(analysis_prompt.format(comments_and_links=analysis_input))
        return result
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None

# Use the function
try:
    # Get the analysis
    analysis = analyze_feedback_by_category(df)

    # Print the results
    print(analysis)

    # Save to file
    with open('feedback_analysis.txt', 'w') as f:
        f.write(analysis)

except Exception as e:
    print(f"Error: {e}")
