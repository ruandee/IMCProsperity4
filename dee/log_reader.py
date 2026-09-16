import json
import re
import sys

def calculate_fill_rate(log_filepath):
    total_executed = 0
    total_ordered = 0
    
    tracker_pattern = re.compile(r"Executed:\s*(\d+)\s*Ordered:\s*(\d+)")

    try:
        with open(log_filepath, 'r') as file:
            log_data = json.load(file)
            ticks = log_data.get('logs', [])
            
            for tick in ticks:
                combined_text = tick.get('sandboxLog', '') + " " + tick.get('lambdaLog', '')
                
                match = tracker_pattern.search(combined_text)
                if match:
                    total_executed += int(match.group(1))
                    total_ordered += int(match.group(2))
                    
    except Exception as e:
        print(f"Error reading log: {e}")
        return

    if total_ordered == 0:
        print("No trade data found.")
        return

    fill_rate = (total_executed / total_ordered) * 100

    print(f"Total Ordered Volume:  {total_ordered}")
    print(f"Total Executed Volume: {total_executed}")
    print(f"Aggregate Fill Rate:   {fill_rate:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python log_reader.py <path_to_log_file>")
    else:
        calculate_fill_rate(sys.argv[1])