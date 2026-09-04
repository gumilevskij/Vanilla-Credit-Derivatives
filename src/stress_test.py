# The code defines a portfolio of European call options.
# It computes portfolio values under increasing volatility shocks (from 1x to 3x base volatility).
# and under increasing gap risk (price drops from 0% to 30%).

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Option class with Black-Scholes pricing
class Option:
    def __init__(self, strike, maturity, notional):
        self.strike = strike
        self.maturity = maturity
        self.notional = notional

    def price(self, spot, vol, r=0.01):
        if self.maturity <= 0:
            return max(0, spot - self.strike) * self.notional
        d1 = (np.log(spot / self.strike) + (r + 0.5 * vol ** 2) * self.maturity) / (vol * np.sqrt(self.maturity))
        d2 = d1 - vol * np.sqrt(self.maturity)
        price = (spot * norm.cdf(d1) - self.strike * np.exp(-r * self.maturity) * norm.cdf(d2)) * self.notional
        return price

# Portfolio setup
portfolio = [
    Option(strike=100, maturity=1.0, notional=1e6),
    Option(strike=110, maturity=0.5, notional=2e6),
    Option(strike=90, maturity=2.0, notional=1.5e6),
]

# Base market parameters
spot = 100
vol = 0.2
r = 0.01

# Stress scenarios functions
def volatility_shock(vol, shock_factor):
    return vol * shock_factor

def gap_risk(spot, gap_size, direction='down'):
    if direction == 'down':
        return spot * (1 - gap_size)
    else:
        return spot * (1 + gap_size)

# Portfolio valuation function
def portfolio_value(portfolio, spot, vol, r):
    return sum(opt.price(spot, vol, r) for opt in portfolio)

# Generate stress scenario ranges
vol_shock_factors = np.linspace(1.0, 3.0, 20)  # From no shock to 3x vol
gap_sizes = np.linspace(0.0, 0.3, 20)          # From no gap to 30% drop

# Calculate portfolio values under volatility shocks
values_vol_shock = [portfolio_value(portfolio, spot, volatility_shock(vol, f), r) for f in vol_shock_factors]

# Calculate portfolio values under gap risk (price drops)
values_gap_risk = [portfolio_value(portfolio, gap_risk(spot, g, 'down'), vol, r) for g in gap_sizes]

# Plots show how portfolio value decreases with rising volatility and price gaps.
plt.figure(figsize=(8, 10))

plt.subplot(2, 1, 1)
plt.plot(vol_shock_factors, values_vol_shock, marker='o')
plt.title('Portfolio Value vs Volatility Shock Factor')
plt.xlabel('Volatility Shock Factor (x base vol)')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)

# Plot portfolio value vs gap size
plt.subplot(2, 1, 2)
plt.plot(gap_sizes * 100, values_gap_risk, marker='o', color='red')
plt.title('Portfolio Value vs Gap Size (Price Drop %)')
plt.xlabel('Gap Size (%)')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print base portfolio value and stressed values at maximum shocks.
base_value = portfolio_value(portfolio, spot, vol, r)
print(f"\nBase Portfolio Value: ${base_value:,.2f}")
print(f"Portfolio Value at 3x Volatility: ${values_vol_shock[-1]:,.2f}")
print(f"Portfolio Value at 30% Price Gap Down: ${values_gap_risk[-1]:,.2f}")
