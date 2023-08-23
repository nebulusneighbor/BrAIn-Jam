import csv

file_path = 'data.csv'
cleaned_file_path = 'data_clean.csv'

# Read the original CSV file
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Clean and write the rows to the cleaned CSV file
with open(cleaned_file_path, 'w', newline='') as cleaned_file:
    writer = csv.writer(cleaned_file)

    for row in rows:
        # Remove commas in the Trigger column
        row[1] = row[1].replace(',', '')

        # Remove any extra commas at the end of the row
        while row and row[-1] == '':
            row.pop()

        # Write the cleaned row to the cleaned file
        writer.writerow(row)

print(f"Cleaned data written to {cleaned_file_path}")
