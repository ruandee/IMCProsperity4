import csv

def filter_trade_data(input_filename, output_filename):
    # 1. Generate the list of allowed products
    strikes = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
    
    # Use a set for O(1) ultra-fast lookups
    allowed_products = {f"VEV_{strike}" for strike in strikes}
    allowed_products.add("VELVETFRUIT_EXTRACT")

    # 2. Open the files (streaming row-by-row)
    with open(input_filename, mode='r', encoding='utf-8') as infile, \
         open(output_filename, mode='w', encoding='utf-8', newline='') as outfile:
        
        # Based on your prompt, the delimiter is a semicolon
        reader = csv.reader(infile, delimiter=';')
        writer = csv.writer(outfile, delimiter=';')
        
        try:
            # Extract and write the header
            header = next(reader)
            writer.writerow(header)
            
            # Find the 'product' column index dynamically (should be 2)
            product_idx = header.index("product")
            
        except StopIteration:
            print("Input file is empty.")
            return
        except ValueError:
            print("Could not find a 'product' column in the header.")
            return

        # 3. Filter and write rows
        rows_kept = 0
        rows_processed = 0
        
        for row in reader:
            rows_processed += 1
            # Ensure the row has enough columns and check if product is allowed
            if len(row) > product_idx and row[product_idx] in allowed_products:
                writer.writerow(row)
                rows_kept += 1
                
            # Optional: Print progress every 500,000 rows
            if rows_processed % 500_000 == 0:
                print(f"Processed {rows_processed:,} rows...")

    print(f"\nDone! Kept {rows_kept:,} out of {rows_processed:,} rows.")
    print(f"Saved to {output_filename}")

# --- Example Usage ---
# Ensure your original file is named 'trades.csv' or change the string below
filter_trade_data('imcprosperity4\dee/round4\combine.csv', 'cleaned_trades.csv')