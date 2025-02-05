def chunk_data(df, chunk_size=5000):
    # Group by Category first
    grouped = df.groupby('Category')
    chunks_by_category = {}

    for category, group in grouped:
        current_chunk = ""
        category_chunks = []

        for _, row in group.iterrows():
            if pd.notna(row['Comments']) and pd.notna(row['att18_@MReplayLink']):
                entry = f"Issue: {str(row['Comments'])}\n"
                entry += f"Link: {str(row['att18_@MReplayLink'])}\n\n"

                if len(current_chunk) + len(entry) > chunk_size:
                    category_chunks.append(current_chunk)
                    current_chunk = entry
                else:
                    current_chunk += entry

        if current_chunk:
            category_chunks.append(current_chunk)

        chunks_by_category[category] = category_chunks

    return chunks_by_category

def analyze_feedback_by_category(df):
    try:
        llm = LlamaLLM(llm_model="llama")
        print("llama", llm.llm_model)

        # Filter data
        df_filtered = df[df['Category'] != 'Unknown'].copy()
        print(f"Records after filtering: {len(df_filtered)}")

        # Get chunks by category
        chunks_by_category = chunk_data(df_filtered)

        # Analysis template for each category
        category_prompt = """Analyze these customer feedback comments for the {category} category.
Group similar issues together and include all relevant QM links in parentheses.
Format the output exactly like this example:
{category} - [Main issue description] (QM_Link1, QM_Link2)

Feedback to analyze:
{feedback}"""

        category_results = {}

        # Process each category
        for category, chunks in chunks_by_category.items():
            category_analysis = []

            for chunk in chunks:
                try:
                    prompt = category_prompt.format(
                        category=category,
                        feedback=chunk
                    )

                    result = llm._call(prompt)
                    if result:
                        category_analysis.append(result)
                except Exception as e:
                    print(f"Error processing chunk for {category}: {e}")

            if category_analysis:
                category_results[category] = " ".join(category_analysis)

        # Final summary prompt
        final_prompt = """Combine these analysis results into a final report using exactly this format:

Web Acct Mgmt - Below are the themes observed for Web Acct Mgmt journey this week are -
Registration & Login - [Group similar issues with their QM links in parentheses]
Profile - [Group similar issues with their QM links in parentheses]
ViewBill & Payment - [Group similar issues with their QM links in parentheses]
TOBR - [Group similar issues with their QM links in parentheses]
View Usage - [Group similar issues with their QM links in parentheses]
Web Sales - [Group similar issues with their QM links in parentheses]
Upper Funnel - [Group similar issues with their QM links in parentheses]

Support & Chat - Support & Chat journey weekly avg rating has improved this week and above the last month avg rating. Some common feedback observed in customer feedback on support & chat are as below.
• Online Support - [List main issues about wait times, chat, service quality]
• Device Unlock - [Group similar issues with their QM links in parentheses]
• Outage Portal - [Group similar issues with their QM links in parentheses]

Rules:
1. Group similar issues together
2. Include all QM links in parentheses after each issue
3. Separate multiple QM links with commas
4. Keep exact format as shown above

Category analyses to combine:
{analyses}"""

        # Combine all category results
        analyses_text = "\n\n".join([f"{cat}:\n{result}" for cat, result in category_results.items()])

        try:
            final_result = llm._call(final_prompt.format(analyses=analyses_text))
            if final_result:
                return final_result
            else:
                return "Error: Final analysis returned None"
        except Exception as e:
            print(f"Error in final analysis: {e}")
            return f"Error in final analysis: {str(e)}"

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error in analysis: {str(e)}"

# Main execution
try:
    print("Starting analysis...")
    analysis = analyze_feedback_by_category(df_summ)

    if analysis:
        if isinstance(analysis, str) and analysis.startswith("Error"):
            print(f"\nAnalysis failed: {analysis}")
        else:
            print("\nAnalysis Results:")
            print(analysis)

            # Save results
            with open('feedback_analysis.txt', 'w', encoding='utf-8') as f:
                f.write(analysis)
            print("Results saved to feedback_analysis.txt")
    else:
        print("\nNo analysis results generated")

except Exception as e:
    print(f"Error in main execution: {str(e)}")
    import traceback
    traceback.print_exc()
