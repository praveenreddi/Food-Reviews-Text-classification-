import pandas as pd

# Read both Excel files
# Replace 'first_file.xlsx' and 'main_file.xlsx' with your actual file names
df_mots = pd.read_excel('first_file.xlsx')
df_main = pd.read_excel('main_file.xlsx')

# Filter rows where active is True
df_mots_filtered = df_mots[df_mots['active'] == True]

# Get the list of applications to search for
applications_to_search = df_mots_filtered['mots_acronym'].tolist()

# Create a mask for matching in either casuer_acronym or primary_application columns
mask = (df_main['casuer_acronym'].isin(applications_to_search)) | \
       (df_main['primary_application'].isin(applications_to_search))

# Filter the main dataframe based on the mask
result_df = df_main[mask]

# Keep only the required columns
result_df = result_df[['ticket_inc', 'month', 'causer_acronym', 'incident_start_time_cst']]

# Remove duplicates based on ticket_inc column
result_df = result_df.drop_duplicates(subset=['ticket_inc'])

# Sort by incident_start_time_cst from past to recent
result_df = result_df.sort_values(by='incident_start_time_cst', ascending=True)

# Save the result to a new Excel file
result_df.to_excel('matched_applications_filtered.xlsx', index=False)

# Print some information about the results
print(f"Total unique tickets found: {len(result_df)}")
print("\nFirst few rows of the sorted data:")
print(result_df.head())

# Created/Modified files during execution:
print("\nCreated/Modified files:")
print("matched_applications_filtered.xlsx")
