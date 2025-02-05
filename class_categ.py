def chunk_analysis(df, max_chunk_size=8000):
    try:
        # Initialize LLM
        llm = LlamaLLM(llm_model="llama")
        print("llama", llm.llm_model)

        # Filter data
        df_filtered = df[df['Category'] != 'Unknown'].copy()
        print(f"Records after filtering: {len(df_filtered)}")

        # Group by category
        grouped = df_filtered.groupby('Category')

        # Initialize results storage
        all_results = []

        # Base prompt template
        base_prompt = """Analyze these customer feedback comments and provide a summary for the following category:

{category}

For each issue, include the QM link in parentheses.
Format each issue as: [Description of issue] (QM Link)

Feedback to analyze:
{feedback}
"""

        # Process each category separately
        for category, group in grouped:
            print(f"Processing category: {category}")

            # Prepare category data
            category_input = ""
            current_chunk = ""
            chunk_count = 1

            for _, row in group.iterrows():
                if pd.notna(row['Comments']) and pd.notna(row['att18_@MReplayLink']):
                    issue_text = f"Issue: {str(row['Comments'])}\nLink: {str(row['att18_@MReplayLink'])}\n\n"

                    # If adding this issue would exceed chunk size, process current chunk
                    if len(current_chunk) + len(issue_text) > max_chunk_size:
                        # Process current chunk
                        prompt = base_prompt.format(
                            category=category,
                            feedback=current_chunk
                        )

                        try:
                            print(f"Processing chunk {chunk_count} for {category}")
                            result = llm(prompt)
                            if result:
                                all_results.append(result)
                        except Exception as e:
                            print(f"Error processing chunk {chunk_count} for {category}: {e}")

                        # Reset chunk
                        current_chunk = issue_text
                        chunk_count += 1
                    else:
                        current_chunk += issue_text

            # Process final chunk if any
            if current_chunk:
                prompt = base_prompt.format(
                    category=category,
                    feedback=current_chunk
                )

                try:
                    print(f"Processing final chunk for {category}")
                    result = llm(prompt)
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"Error processing final chunk for {category}: {e}")

        # Combine all results
        if all_results:
            # Final summary prompt
            final_prompt = """Combine and summarize these analysis results into a final report using this exact format:

Web Acct Mgmt - Below are the themes observed for Web Acct Mgmt journey this week:

Registration & Login
- [List main issues with QM links]

Profile
- [List main issues with QM links]

ViewBill & Payment
- [List main issues with QM links]

TOBR
- [List main issues with QM links]

View Usage
- [List main issues with QM links]

Web Sales
- [List main issues with QM links]

Upper Funnel
- [List main issues with QM links]

Support & Chat
• Online Support
- [List issues about wait times, chat, service quality]
• Device Unlock
- [List main issues with QM links]
• Outage Portal
- [List main issues with QM links]

Analysis results to combine:
{results}
"""

            try:
                final_result = llm(final_prompt.format(results="\n\n".join(all_results)))
                return final_result
            except Exception as e:
                print(f"Error in final summary: {e}")
                return "\n\n".join(all_results)
        else:
            return "No results generated from analysis"

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error in analysis: {str(e)}"

# Main execution
try:
    print("Starting analysis...")
    analysis = chunk_analysis(df_summ)

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
