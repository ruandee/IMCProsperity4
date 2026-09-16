import pandas as pd
import numpy as np
import statsmodels.api as sm

def calculate_ou_parameters():
    # 1. Load the historical data 
    df0 = pd.read_csv('imcprosperity4/dee/round3/data/prices_round_3_day_0.csv', sep=';')
    df1 = pd.read_csv('imcprosperity4/dee/round3/data/prices_round_3_day_1.csv', sep=';')
    df2 = pd.read_csv('imcprosperity4/dee/round3/data/prices_round_3_day_2.csv', sep=';')
    df = pd.concat([df0, df1, df2])

    # 2. Filter and sort VELVETFRUIT_EXTRACT
    velvet = df[df['product'] == 'VELVETFRUIT_EXTRACT'].copy()
    velvet['continuous_time'] = velvet['day'] * 1000000 + velvet['timestamp']
    velvet = velvet.sort_values('continuous_time').reset_index(drop=True)

    # 3. Extract prices and calculate dS
    S = velvet['mid_price'].dropna().values
    S_prev = S[:-1]
    dS = S[1:] - S_prev

    # 4. Linear Regression: dS = a + b * S_prev + epsilon
    X = sm.add_constant(S_prev)
    model = sm.OLS(dS, X).fit()

    a = model.params[0]
    b = model.params[1]

    # 5. Define Time Step (dt)
    # 10,000 timestamps per day. Annualized using 252 trading days.
    dt_years = 1.0 / (252.0 * 10000.0)

    # 6. Calculate OU Parameters
    theta = -a / b
    kappa = -np.log(1 + b) / dt_years
    
    # Sigma is the standard deviation of the residuals scaled by sqrt(dt)
    sigma = np.std(model.resid) / np.sqrt(dt_years)

    print(f"--- ORNSTEIN-UHLENBECK PARAMETERS (Annualized) ---")
    print(f"Theta (Mean):   {theta:.2f}")
    print(f"Kappa (Speed):  {kappa:.2f}")
    print(f"Sigma (Vol):    {sigma:.2f}")

calculate_ou_parameters()