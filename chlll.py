def analyze_detailed_issues(result_dict):
    prompt = """
    Analyze all feedback chunks and provide a comprehensive summary in exactly this format:

    Web Acct Mgmt - Below are the themes observed for Web Acct Mgmt journey this week are -

    • Registration & Login - [List specific issues with exact error messages] (group related QM links in parentheses, separated by commas)

    • Profile - [List specific issues with exact error messages] (group related QM links in parentheses)

    • ViewBill & Payment - [List specific issues with exact error messages] (group related QM links in parentheses)

    • TOBR - [List specific issues with exact error messages] (group related QM links in parentheses)

    • View Usage - [List specific issues with exact error messages] (group related QM links in parentheses)

    Web Sales - [List specific issues with exact error messages and their QM links]

    Support & Chat - Support & Chat journey weekly avg rating [include rating details if available]. Some common feedback observed in customer feedback on support & chat are as below:
    • Online Support - [List specific feedback points separated by commas]
    • Device Unlock - [List specific issues with QM links]
    • Outage Portal - [List specific feedback]

    Important formatting rules:
    1. Maintain exact category headers (Web Acct Mgmt, Web Sales, Support & Chat)
    2. Include detailed error messages in quotes exactly as reported
    3. Group related QM links within parentheses
    4. Use bullet points (•) for subcategories
    5. Separate multiple issues within a category with commas
    6. Include all relevant QM links for each issue
    7. Maintain the exact hierarchical structure

    Results to analyze:
    {text}
    """

    try:
        result_text = result_dict.get('result', '')
        formatting_prompt = prompt.format(text=result_text)

        formatted_result = llm(formatting_prompt)

        print(formatted_result)

        return formatted_result

    except Exception as e:
        print(f"Error in formatting: {e}")
        return None

# Usage
try:
    query = """
    Analyze all feedback chunks and provide a comprehensive summary of:
    1. Web Account Management issues and their QM links
    2. Web Sales issues and their QM links
    3. Support & Chat feedback
    Group similar issues together and include all relevant QM links.
    """

    result = chain({"query": query})
    analysis = analyze_detailed_issues(result)

except Exception as e:
    print(f"Error: {e}")
