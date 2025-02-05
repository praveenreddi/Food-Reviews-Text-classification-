def analyze_feedback_by_category(df):
    try:
        # Initialize LLM
        callback_handler = MyCustomHandler()
        llm = LlamaLLM(llm_model="llama", callbacks=[callback_handler])

        # Filter data
        df_filtered = df[df['Category'] != 'Unknown'].copy()
        print(f"Records after filtering: {len(df_filtered)}")

        # Prepare the input data
        analysis_input = ""
        for _, row in df_filtered.iterrows():
            category = row.get('Category', '')
            comment = row.get('Comments', '')
            qm_link = row.get('att18_@MReplayLink', '')

            if pd.notna(comment) and pd.notna(qm_link):
                analysis_input += f"Category: {category}\n"
                analysis_input += f"Comment: {str(comment)}\n"
                analysis_input += f"QM Link: {str(qm_link)}\n\n"

        # Create the prompt
        analysis_prompt = """Analyze these customer feedback comments and provide a summary in exactly this format:

Web Acct Mgmt - Below are the themes observed for Web Acct Mgmt journey this week:

Registration & Login
- [List issues with QM links in parentheses]

Profile
- [List issues with QM links in parentheses]

ViewBill & Payment
- [List issues with QM links in parentheses]

TOBR
- [List issues with QM links in parentheses]

View Usage
- [List issues with QM links in parentheses]

Web Sales
- [List issues with QM links in parentheses]

Upper Funnel
- [List issues with QM links in parentheses]

Support & Chat
• Online Support
- [List issues about wait times, chat, service quality]
• Device Unlock
- [List issues with QM links in parentheses]
• Outage Portal
- [List issues with QM links in parentheses]

Analyze these comments:
{comments_and_links}
"""

        # Make the LLM call
        try:
            result = llm._call(analysis_prompt.format(comments_and_links=analysis_input))
            if result is None:
                raise ValueError("LLM returned None response")
            return result
        except Exception as llm_error:
            print(f"LLM Error: {llm_error}")
            # Fallback to basic analysis
            return f"Error in LLM processing: {str(llm_error)}"

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
try:
    # Read the data
    data_path2 = "abfss://opsdashboards@blackbirdproddatastore.dfs.core.windows.net/VOC/OL_data/OL_results/VOCcat_results.csv"
    df_cp = pd.read_csv(data_path2, storage_options={"sas_token": base64.b64decode(sas_token).decode()})
    df_summ = df_cp.copy()

    # Run analysis
    print("Starting analysis...")
    analysis = analyze_feedback_by_category(df_summ)

    if analysis and not isinstance(analysis, str) or not analysis.startswith("Error"):
        print("\nAnalysis Results:")
        print(analysis)

        # Save results
        with open('feedback_analysis.txt', 'w', encoding='utf-8') as f:
            f.write(analysis)
        print("Results saved to feedback_analysis.txt")
    else:
        print(f"\nAnalysis failed: {analysis}")

except Exception as e:
    print(f"Error in main execution: {str(e)}")
    import traceback
    traceback.print_exc()
