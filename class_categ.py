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

# Save the result to a new Excel file
result_df.to_excel('matched_applications.xlsx', index=False)

# Print some information about the results
print(f"Total applications to search: {len(applications_to_search)}")
print(f"Total matches found: {len(result_df)}")

# Created/Modified files during execution:
print("Created/Modified files:")
print("matched_applications.xlsx")
