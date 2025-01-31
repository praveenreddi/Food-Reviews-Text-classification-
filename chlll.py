def create_qa_chain():
    # Increase the number of relevant chunks retrieved
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,  # Increase this number to get more relevant chunks
            "fetch_k": 20  # Fetch more documents before selecting top k
        }
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use 'stuff' to include all retrieved documents
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# Modified prompt for better coverage
def analyze_detailed_issues(result_dict):
    prompt = """
    Analyze ALL the provided feedback thoroughly and create a comprehensive summary.
    You MUST check and include issues for ALL of these categories if they exist:

    1. Web Acct Mgmt (including):
       - Registration & Login
       - Profile
       - ViewBill & Payment
       - TOBR
       - View Usage

    2. Web Sales

    3. Support & Chat (including):
       - Online Support
       - Device Unlock
       - Outage Portal

    Format the output exactly like this:

    Web Acct Mgmt - Below are the themes observed for Web Acct Mgmt journey this week are -

    • Registration & Login - [List ALL specific issues with exact error messages] (include ALL related QM links)

    [Continue with other categories...]

    Important rules:
    1. DO NOT skip any category that has issues
    2. Include ALL QM links for each issue
    3. Check thoroughly for issues in each category
    4. Use exact error messages in quotes
    5. If a category truly has no issues, only then mark it as "No issues reported"

    Results to analyze:
    {text}
    """

    try:
        result_text = result_dict.get('result', '')
        # Ensure we're getting all the content
        if isinstance(result_text, list):
            result_text = ' '.join(result_text)

        formatting_prompt = prompt.format(text=result_text)

        # Increase max tokens if needed
        formatted_result = llm(formatting_prompt, max_tokens=4000)

        print(formatted_result)
        return formatted_result

    except Exception as e:
        print(f"Error in formatting: {e}")
        return None

# Usage with modified query
try:
    query = """
    Provide a comprehensive analysis of ALL customer feedback and issues across ALL categories:
    - Web Account Management (including all subcategories)
    - Web Sales
    - Support & Chat

    Include ALL relevant issues and their corresponding QM links.
    Do not skip any category that has reported issues.
    """

    # Increase the number of relevant chunks
    result = chain({"query": query})
    analysis = analyze_detailed_issues(result)

except Exception as e:
    print(f"Error: {e}")
