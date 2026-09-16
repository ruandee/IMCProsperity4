import pandas as pd
import json
import matplotlib.pyplot as plt
import io
import ast
import os
import sys

def parse_prosperity_log(filepath):
    print(f"Reading and parsing {filepath}...")
    activities_csv = []
    trade_json_lines = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    mode = None
    for line in lines:
        if "Activities log:" in line:
            mode = "activities"
            continue
        elif "Trade History:" in line:
            mode = "trades"
            continue
        elif line.startswith("Sandbox logs:"):
            mode = "sandbox" # Skip all lambda and sandbox logs
            continue
            
        if mode == "activities":
            # Only append valid CSV rows
            if line.strip() and not line.strip().startswith("{") and not line.strip().startswith("}"):
                activities_csv.append(line.strip())
        elif mode == "trades":
            if line.strip():
                trade_json_lines.append(line)
                
    # 1. Parse Activities
    csv_data = "\n".join(activities_csv)
    activities_df = pd.read_csv(io.StringIO(csv_data), sep=';')
    
    # 2. Parse Trades
    trade_str = "".join(trade_json_lines)
    trades_df = pd.DataFrame()
    
    if trade_str:
        trade_str = trade_str.replace(",\n  }", "\n  }").replace(",\n]", "\n]")
        try:
            trades = ast.literal_eval(trade_str)
            trades_df = pd.DataFrame(trades)
        except Exception as e:
            try:
                trades = json.loads(trade_str)
                trades_df = pd.DataFrame(trades)
            except Exception as e2:
                print(f"Warning: Could not parse trades cleanly. Error: {e2}")

    return activities_df, trades_df

def plot_target_graphs(activities_df, trades_df, log_name, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    symbols = activities_df['product'].unique()
    
    # Match any VEV, VELVETFRUIT, HYDROGEL, or HYPERGEL products
    target_keywords = ["PEBBLES_XL", "PEBBLES_L","PEBBLES_M","PEBBLES_S","PEBBLES_XS"]
    target_symbols = [
        sym for sym in symbols 
        if any(keyword in str(sym) for keyword in target_keywords)
    ]
    
    if not target_symbols:
        print("No target symbols found in the log.")
        return

    valid_symbols = []
    symbol_trade_data = {}
    
    # --- 1. Pre-filter products to only keep those we traded ---
    for symbol in target_symbols:
        my_buys, my_sells, bot_trades = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if not trades_df.empty and 'symbol' in trades_df.columns:
            df_trades = trades_df[trades_df['symbol'] == symbol].copy()
            if not df_trades.empty:
                df_trades.sort_values('timestamp', inplace=True)
                my_buys = df_trades[df_trades['buyer'] == 'SUBMISSION']
                my_sells = df_trades[df_trades['seller'] == 'SUBMISSION']
                bot_trades = df_trades[(df_trades['buyer'] != 'SUBMISSION') & (df_trades['seller'] != 'SUBMISSION')]
                
        # If we made 0 buys and 0 sells, skip it!
        if my_buys.empty and my_sells.empty:
            print(f"Skipping {symbol} - 0 trades made by us.")
        else:
            valid_symbols.append(symbol)
            symbol_trade_data[symbol] = (my_buys, my_sells, bot_trades)
            
    num_plots = len(valid_symbols)
    if num_plots == 0:
        print("\nNo symbols with executed trades were found to plot.")
        return

    print(f"\nGenerating combined graph for {num_plots} traded products...")
    
    # --- 2. Create one large figure dynamically sized based on the number of valid symbols ---
    # 6 inches of height per product
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 6 * num_plots), sharex=True)
    fig.suptitle(f'Combined Trade Executions ({log_name})', fontsize=20, fontweight='bold', y=0.98)
    
    # Ensure axes is iterable even if there is only 1 valid symbol
    if num_plots == 1:
        axes = [axes]

    # --- 3. Plot only the 'Combined Picture' graph for each valid symbol ---
    for idx, symbol in enumerate(valid_symbols):
        ax = axes[idx]
        my_buys, my_sells, bot_trades = symbol_trade_data[symbol]
        
        df_act = activities_df[activities_df['product'] == symbol].copy()
        if 'timestamp' in df_act.columns:
            df_act.sort_values('timestamp', inplace=True)
            
            # Thin lines (linewidth=0.5)
            ax.plot(df_act['timestamp'], df_act['mid_price'], label='Mid Price', color='black', alpha=0.8, linestyle='--', linewidth=0.5)
            if 'ask_price_1' in df_act.columns:
                ax.plot(df_act['timestamp'], df_act['ask_price_1'], label='Ask Price 1', color='red', alpha=0.5, linewidth=0.5)
            if 'bid_price_1' in df_act.columns:
                ax.plot(df_act['timestamp'], df_act['bid_price_1'], label='Bid Price 1', color='green', alpha=0.5, linewidth=0.5)
                
        # Triangles for Buys and Sells overlayed
        if not my_buys.empty:
            ax.scatter(my_buys['timestamp'], my_buys['price'], marker='^', color='green', s=150, zorder=5, edgecolor='black', label='I Bought')
        if not my_sells.empty:
            ax.scatter(my_sells['timestamp'], my_sells['price'], marker='v', color='red', s=150, zorder=5, edgecolor='black', label='I Sold')
        if not bot_trades.empty:
            ax.scatter(bot_trades['timestamp'], bot_trades['price'], marker='o', color='blue', s=20, alpha=0.2, zorder=3, label='Bot Trades')
            
        ax.set_title(f'Product: {symbol}', fontsize=14)
        ax.set_ylabel('Price')
        
        # Only put the X-label on the very bottom graph
        if idx == num_plots - 1:
            ax.set_xlabel('Timestamp')
            
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Save to a single file with Log Name
    plt.tight_layout()
    # Adjust spacing so the title doesn't overlap the first graph
    fig.subplots_adjust(top=0.95)
    
    output_path = os.path.join(output_dir, f"all_trades_combined_{log_name}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved master chart: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python log_visualiser.py <path_to_log_file>")
        sys.exit(1)
        
    log_file_path = sys.argv[1]
    
    try:
        activities_df, trades_df = parse_prosperity_log(log_file_path)
        log_dir = os.path.dirname(os.path.abspath(log_file_path))
        plot_dir = os.path.join(log_dir, "plots")
        
        # Extract the base name of the log file
        log_name = os.path.splitext(os.path.basename(log_file_path))[0]
        
        plot_target_graphs(activities_df, trades_df, log_name, output_dir=plot_dir)
        print(f"\nSuccess! Check the '{plot_dir}' folder.")
    except FileNotFoundError:
        print(f"Error: Could not find the file at '{log_file_path}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")