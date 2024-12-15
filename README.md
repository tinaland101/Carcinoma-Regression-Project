# Mathplotlib_Challenge

# create variable for reading cvs from computer
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

# Paths to your data files
mouse_metadata_path = "/Users/christinaland/Downloads/Starter_Code-9/Pymaceuticals/data/Mouse_metadata.csv"
study_results_path = "/Users/christinaland/Downloads/Starter_Code-9/Pymaceuticals/data/Study_results.csv"

# Read the CSV files into DataFrames
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)

# Merge the data on 'Mouse ID'
merged_data = pd.merge(mouse_metadata, study_results, on="Mouse ID")

# Specify the desired column order
desired_column_order = [
    'Mouse ID', 'Timepoint', 'Tumor Volume (mm3)', 'Metastatic Sites',
    'Drug Regimen', 'Sex', 'Age_months', 'Weight (g)'
]

# Reorganize the columns
data_merged_col = merged_data[desired_column_order]

# Sort by 'Timepoint' in ascending order
data_merged_col_sorted = data_merged_col.sort_values(by="Timepoint")

# Display the sorted data
print(data_merged_col_sorted.head())


# number of mice
mouse_sum = data_merged_col["Mouse ID"].nunique()
print(mouse_sum)

hould be uniquely identified by Mouse ID and Timepoint
# Get the duplicate mice by ID number that shows up for Mouse ID and Timepoint
duplicate_mice = data_merged_col[data_merged_col.duplicated(subset=["Mouse ID", "Timepoint"], keep=False)]

duplicate_mice_ids = duplicate_mice["Mouse ID"].unique()


print(duplicate_mice_ids, duplicate_mice_ids.dtype)
duplicates_df = data_merged_col[data_merged_cleaned.duplicated(subset=['Mouse ID', 'Timepoint'])]
print(duplicates)

data_merged = pd.merge(mouse_metadata, study_results, on="Mouse ID")


duplicates = data_merged[data_merged.duplicated(subset=["Mouse ID", "Timepoint"], keep="first")]



data_cleaned = data_merged.drop_duplicates(subset="Mouse ID", keep="first")





print("\nCleaned DataFrame:")
print(data_cleaned.head())


unique_mice_count = data_cleaned['Mouse ID'].nunique()
print(unique_mice_count)

# Group the merged data by 'Drug Regimen'
grouped_data = data_merged.groupby('Drug Regimen')['Tumor Volume (mm3)']

# Calculate summary statistics
mean_tumor_volume = grouped_data.mean()
median_tumor_volume = grouped_data.median()
variance_tumor_volume = grouped_data.var()
std_dev_tumor_volume = grouped_data.std()
sem_tumor_volume = grouped_data.sem()

# Create a summary DataFrame by combining all the calculated statistics
summary_stats = pd.DataFrame({
    'Mean Tumor Volume': mean_tumor_volume,
    'Median Tumor Volume': median_tumor_volume,
    'Tumor Volume Variance': variance_tumor_volume,
    'Tumor Volume Std. Dev.': std_dev_tumor_volume,
    'Tumor Volume Std. Err.': sem_tumor_volume
})

# Display the summary statistics table
print(summary_stats)

import pandas as pd
# Count the number of rows (Mouse ID/Timepoints) for each drug regimen
rows_per_drug_regimen = data_merged.groupby('Drug Regimen').size()

# Sort the values in descending order
rows_per_drug_regimen = rows_per_drug_regimen.sort_values(ascending=False)

# Create the bar plot using Pandas plot function
rows_per_drug_regimen.plot(kind='bar', figsize=(10, 6), color='skyblue')

# Adding labels and title
plt.title('Total Number of Rows (Mouse ID/Timepoints) per Drug Regimen')
plt.xlabel('Drug Regimen')
plt.ylabel('Number of Rows (Mouse ID/Timepoints)')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()  # Ensure the labels fit in the plot
plt.show()
import matplotlib.pyplot as plt
# Count the number of rows (Mouse ID/Timepoints) for each drug regimen
rows_per_drug_regimen = data_merged.groupby('Drug Regimen').size()

# Sort the values in descending order
rows_per_drug_regimen = rows_per_drug_regimen.sort_values(ascending=False)

# Generate the bar plot using Pandas
rows_per_drug_regimen.plot(kind='bar', color='blue', figsize=(10, 6))



plt.ylabel('# of Observed Mouse Timepoints')
plt.xlabel('Drug Regimen')
plt.xticks(rotation=45)
plt.tight_layout()  # Ensure the labels fit in the plot
plt.show()
import pandas as pd
 
 #Get the unique mice and their gender
unique_mice_gender = data_merged.drop_duplicates(subset='Mouse ID')['Sex']

# Count the number of unique female and male mice
gender_counts = unique_mice_gender.value_counts()

# Create a pie chart using Pandas
gender_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(7, 7), startangle=90, colors=['lightblue', 'pink'])



# Show the plot
plt.ylabel('')  # Hide the y-label (it is redundant for a pie chart)
plt.tight_layout()  # Ensure the labels fit within the plot space
plt.show()

import matplotlib.pyplot as plt

# Assuming 'data_merged' is already created and merged from the previous steps

# Count the number of female and male mice in the study
gender_count = data_merged['Sex'].value_counts()

# Create the pie chart using Matplotlib's pyplot methods
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])

# Show the plot
plt.show()
import pandas as pd

# Assuming 'data_merged' is already loaded and merged with all columns.

# Group by Drug Regimen and get the final tumor volume for each mouse
last_timepoints = data_merged.groupby('Mouse ID')['Timepoint'].max()
last_timepoints_df = pd.merge(last_timepoints, data_merged, on=['Mouse ID', 'Timepoint'], how='left')

# Filter for the treatment regimens of interest
treatment_regimens = ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin']
filtered_data = last_timepoints_df[last_timepoints_df['Drug Regimen'].isin(treatment_regimens)]

# Create a dictionary to hold potential outliers for each treatment regimen
outliers = {}

# Loop through each treatment regimen to find outliers
for regimen in treatment_regimens:
    # Filter the data for the current regimen
    regimen_data = filtered_data[filtered_data['Drug Regimen'] == regimen]['Tumor Volume (mm3)']
    
    # Calculate the quartiles and IQR
    Q1 = regimen_data.quantile(0.25)
    Q3 = regimen_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify potential outliers
    potential_outliers = regimen_data[(regimen_data < lower_bound) | (regimen_data > upper_bound)]
    
    # Store the outliers for this regimen
    outliers[regimen] = potential_outliers

# Display the potential outliers for each regimen
for regimen, outlier_data in outliers.items():
    print(f"{regimen}'s potential outliers: {outlier_data}\n")
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data_merged' is already loaded and merged with all columns.

# Group by Drug Regimen and get the final tumor volume for each mouse
last_timepoints = data_merged.groupby('Mouse ID')['Timepoint'].max()
last_timepoints_df = pd.merge(last_timepoints, data_merged, on=['Mouse ID', 'Timepoint'], how='left')

# Filter for the treatment regimens of interest
treatment_regimens = ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin']
filtered_data = last_timepoints_df[last_timepoints_df['Drug Regimen'].isin(treatment_regimens)]

# Create a dictionary to hold potential outliers for each treatment regimen
outliers = {}

# Create a list to hold tumor volume data for each regimen (for the box plot)
tumor_volumes = []

# Loop through each treatment regimen to find outliers and prepare data for the box plot
for regimen in treatment_regimens:
    # Filter the data for the current regimen
    regimen_data = filtered_data[filtered_data['Drug Regimen'] == regimen]['Tumor Volume (mm3)']
    
    # Calculate the quartiles and IQR
    Q1 = regimen_data.quantile(0.25)
    Q3 = regimen_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify potential outliers
    potential_outliers = regimen_data[(regimen_data < lower_bound) | (regimen_data > upper_bound)]
    
    # Store the outliers for this regimen
    outliers[regimen] = potential_outliers
    
    # Append tumor volume data for box plot
    tumor_volumes.append(regimen_data)

# Display the potential outliers for each regimen
for regimen, outlier_data in outliers.items():
    print(f"{regimen}'s potential outliers: {outlier_data}\n")

# Create the box plot
plt.figure(figsize=(10, 6))
plt.boxplot(tumor_volumes, labels=treatment_regimens, patch_artist=True)

# Adding labels and title
plt.title('Distribution of Tumor Volumes for Each Treatment Group')
plt.xlabel('Drug Regimen')
plt.ylabel('Tumor Volume (mm3)')

# Show the plot
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data_merged' is already loaded and merged with all columns.

# Filter the data for a single mouse treated with Capomulin
mouse_id = 'l509'  # Example Mouse ID for a mouse treated with Capomulin
capomulin_data = data_merged[(data_merged['Drug Regimen'] == 'Capomulin') & (data_merged['Mouse ID'] == mouse_id)]

# Create the plot of Tumor Volume vs. Timepoint (line plot without grid lines and points)
plt.figure(figsize=(8, 6))

# Plot with a solid line, but no points (no markers)
plt.plot(capomulin_data['Timepoint'], capomulin_data['Tumor Volume (mm3)'], color='b', linestyle='-', marker='', linewidth=2)

# Add titles and labels
plt.title(f'Tumor Volume vs. Timepoint for Mouse {mouse_id} (Capomulin Treatment)', fontsize=14)
plt.xlabel('Timepoint (Days)', fontsize=12)
plt.ylabel('Tumor Volume (mm3)', fontsize=12)

# Remove the grid
plt.grid(False)

# Display the plot
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data_merged' is already loaded and merged with all columns.

# Filter the data for the Capomulin regimen
capomulin_data = data_merged[data_merged['Drug Regimen'] == 'Capomulin']

# Group by 'Mouse ID' and calculate the average tumor volume for each mouse
average_tumor_volume = capomulin_data.groupby('Mouse ID')['Tumor Volume (mm3)'].mean()

# Merge the average tumor volume with the mouse weights (using 'Mouse ID')
mouse_weights = capomulin_data[['Mouse ID', 'Weight (g)']].drop_duplicates()
merged_data = pd.merge(average_tumor_volume, mouse_weights, on='Mouse ID')

# Create a scatter plot of mouse weight vs. average tumor volume
plt.figure(figsize=(8, 6))
plt.scatter(merged_data['Weight (g)'], merged_data['Tumor Volume (mm3)'], color='b', edgecolor='black', alpha=0.7)

# Add titles and labels
plt.title('Mouse Weight vs. Average Tumor Volume for Capomulin Regimen', fontsize=14)
plt.xlabel('Weight (g)', fontsize=12)
plt.ylabel('Average Tumor Volume (mm3)', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Assuming 'data_merged' is already loaded and merged with all columns.

# Filter the data for the Capomulin regimen
capomulin_data = data_merged[data_merged['Drug Regimen'] == 'Capomulin']

# Group by 'Mouse ID' and calculate the average tumor volume for each mouse
average_tumor_volume = capomulin_data.groupby('Mouse ID')['Tumor Volume (mm3)'].mean()

# Merge the average tumor volume with the mouse weights (using 'Mouse ID')
mouse_weights = capomulin_data[['Mouse ID', 'Weight (g)']].drop_duplicates()
merged_data = pd.merge(average_tumor_volume, mouse_weights, on='Mouse ID')

# Calculate the Pearson correlation coefficient
correlation = merged_data['Weight (g)'].corr(merged_data['Tumor Volume (mm3)'])
print(f"Pearson correlation coefficient: {correlation}")

# Perform linear regression to get the slope and intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_data['Weight (g)'], merged_data['Tumor Volume (mm3)'])

# Create the regression line
regression_line = slope * merged_data['Weight (g)'] + intercept

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(merged_data['Weight (g)'], merged_data['Tumor Volume (mm3)'], color='b', edgecolor='black', alpha=0.7)

# Plot the regression line
plt.plot(merged_data['Weight (g)'], regression_line, color='r', linewidth=2)

# Add titles and labels
plt.title('Mouse Weight vs. Average Tumor Volume for Capomulin Regimen', fontsize=14)
plt.xlabel('Weight (g)', fontsize=12)
plt.ylabel('Average Tumor Volume (mm3)', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()

# Print the linear regression results
print(f"Linear Regression Model: y = {slope:.2f}x + {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")
