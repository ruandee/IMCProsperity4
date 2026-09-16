import pandas as pd
import numpy as np

def analyze_stickiness(file_path, product):
    # Load data
    df = pd.read_csv(file_path, sep=';')
    df = df[df['product'] == product].copy()
    
    # 1. Transition Probability (Tick-to-Tick)
    df['mid_price_change'] = df['mid_price'].diff()
    
    # Drop the first NaN value
    changes = df['mid_price_change'].dropna()
    
    p_sticky = (changes == 0).mean()
    p_up = (changes > 0).mean()
    p_down = (changes < 0).mean()
    
    print(f"--- {product} Stickiness Profile ---")
    print(f"Probability of staying at current price: {p_sticky:.2%}")
    print(f"Probability of moving UP: {p_up:.2%}")
    print(f"Probability of moving DOWN: {p_down:.2%}")
    
    # 2. Average Residence Time
    # We create a new 'regime_id' every time the price changes
    df['is_new_price'] = (df['mid_price'] != df['mid_price'].shift(1)).astype(int)
    df['price_regime_id'] = df['is_new_price'].cumsum()
    
    # Count how many timestamps each regime lasted
    residence_times = df.groupby('price_regime_id').size()
    
    # Filter out the last regime since it might be cut off by the end of the day
    residence_times = residence_times[:-1]
    
    print(f"\nAverage Residence Time: {residence_times.mean():.2f} timestamps")
    print(f"Median Residence Time: {residence_times.median():.2f} timestamps")
    print(f"Max Residence Time observed: {residence_times.max()} timestamps")
    print("-" * 40)

# Run the analysis
analyze_stickiness('prices_round_0_day_-1.csv', 'EMERALDS')
analyze_stickiness('prices_round_0_day_-1.csv', 'TOMATOES')