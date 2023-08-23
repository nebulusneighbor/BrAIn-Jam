import csv

file_path = 'data_clean.csv'
cleaned_file_path = 'data_clean_2x.csv'

def clean_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        previous_trigger = None
        for row in reader:
            # Check the trigger column (index 1) and extract the name if it's in the specific format
            trigger = row[1] if len(row) > 1 else None
            if trigger and trigger.startswith("("):
                # Extract the trigger name without parentheses, apostrophe, and the timestamp
                trigger = trigger.split("' ")[0][2:]
                
                # If the trigger is different from the previous one or it's the first row (header), write the row
                if trigger != previous_trigger or reader.line_num == 1:
                    writer.writerow([row[0], trigger] + row[2:])
                
                # Update the previous trigger
                previous_trigger = trigger
            else:
                writer.writerow(row)

clean_csv(file_path, cleaned_file_path)
