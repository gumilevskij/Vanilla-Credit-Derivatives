import QuantLib as ql
import numpy as np
from scipy.stats import norm, expon
import matplotlib.pyplot as plt

# Set evaluation date
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# 1. Build hazard curves for each entity (flat hazard rates)
def build_flat_hazard_curve(hazard_rate, today):
    hazard_rate_quote = ql.SimpleQuote(hazard_rate)
    hazard_rate_handle = ql.QuoteHandle(hazard_rate_quote)
    return ql.FlatHazardRate(today, hazard_rate_handle, ql.Actual365Fixed())

# 3. Simulate correlated default times using Gaussian copula
def simulate_default_times(hazard_rates, corr_matrix, n_simulations, maturity):
    n_entities = len(hazard_rates)
    #mean = np.zeros(n_entities)
    L = np.linalg.cholesky(corr_matrix)
    default_times = np.zeros((n_simulations, n_entities))

    for i in range(n_simulations):
        z = np.random.normal(size=n_entities)
        correlated_normals = L @ z
        for j in range(n_entities):
            u = norm.cdf(correlated_normals[j])
            # Convert uniform to exponential default time
            default_times[i, j] = expon.ppf(u, scale=1/hazard_rates[j])
    return default_times

# 4. Calculate tranche losses
def tranche_loss(default_times, maturity, notionals, recovery_rate, attachment, detachment):
    n_simulations, n_entities = default_times.shape
    tranche_losses = np.zeros(n_simulations)
    tranche_notional = detachment - attachment

    for i in range(n_simulations):
        losses = np.array([notionals[j] * (1 - recovery_rate) if default_times[i, j] <= maturity else 0 for j in range(n_entities)])
        portfolio_loss = losses.sum() / notionals.sum()  # normalized portfolio loss (0 to 1)
        # Tranche loss calculation (normalized)
        loss = max(0, min(portfolio_loss - attachment, tranche_notional))
        tranche_losses[i] = loss / tranche_notional
    return tranche_losses

def loss_probability(rho,attachment,detachment,t,default_times):
    # Gaussian copula simulation for correlated defaults
    common_factor = np.random.normal(0, 1, num_simulations)
    idiosyncratic = np.random.normal(0, 1, (num_simulations, n_entities))
    
    asset_values = np.sqrt(rho) * common_factor[:, None] + np.sqrt(1 - rho) * idiosyncratic
    
    default_threshold = norm.ppf(default_prob)
    defaults = (asset_values < default_threshold).astype(int)
    
    portfolio_loss = defaults.mean(axis=1)  # fraction of assets defaulted
    
    I1 = detachment > portfolio_loss
    I2 = portfolio_loss >= attachment
    loss_prob = np.mean(I1*I2)
    # TODO implement default_times
    loss_prob = []
    for i in range(n_simulations):
        times = default_times[i]
        for j in range(n_entities):
            common_factor = np.random.normal(0, 1, num_simulations)
            idiosyncratic = np.random.normal(0, 1, (num_simulations, n_entities))
            
            asset_values = np.sqrt(rho) * common_factor[:, None] + np.sqrt(1 - rho) * idiosyncratic
            
            default_threshold = norm.ppf(default_prob)
            defaults = (asset_values < default_threshold).astype(int)
            
            portfolio_loss = defaults.mean(axis=1)  # fraction of assets defaulted
            
            I1 = detachment > portfolio_loss
            I2 = portfolio_loss >= attachment
            if t > times[j]:
                loss_prob.append(np.mean(I1*I2))
            else:
                loss_prob.append(0)
                
    loss_prob = np.mean(loss_prob)           
    return loss_prob

# Example portfolio data
n_entities = 5
hazard_rates = [0.01, 0.015, 0.02, 0.012, 0.018]  # annual hazard rates
notionals = np.array([10e6, 10e6, 10e6, 10e6, 10e6])  # notionals per entity
recovery_rate = 0.4
maturity = 5  # years
discount_rate = 0.05
num_simulations = 100
rho = 0.2 # correlation
default_prob = 0.05  # individual default probability


# Build hazard curves (not directly used in simulation but illustrative)
hazard_curves = [build_flat_hazard_curve(hr, today) for hr in hazard_rates]

# 2. Define correlation matrix (example: 0.3 correlation between entities)
corr_matrix = np.full((n_entities, n_entities), 0.3)
np.fill_diagonal(corr_matrix, 1.0)

n_simulations = 10000
default_times = simulate_default_times(hazard_rates, corr_matrix, n_simulations, maturity)

# Define tranche attachment and detachment points (e.g., 3% to 7%)
attachments = [0,0.03,0.07,0.12,0.25]
detachments = [0.03,0.07,0.12,0.25,1.0]
tranche_name = ['Equity','Junior Mezzanine','Senior Mezzanine','Senior','Super Senior']
n_tranches = len(attachments)
loss_levels = np.linspace(0, 1, 100)  # Loss from 0% to 100%

i = 0
expected_tranche_loss = np.empty(n_tranches); spread =  np.empty(n_tranches)
for attachment,detachment in zip(attachments,detachments):
    tranche_losses = tranche_loss(default_times, maturity, notionals, recovery_rate, attachment, detachment)
    
    # 5. Estimate expected tranche loss (proxy for tranche value)
    expected_tranche_loss[i] = np.mean(tranche_losses)
    print(f"Expected tranche loss [{100*attachment:.0f}% - {100*detachment:.0f}%]: {expected_tranche_loss[i]:.3e}")

    # Calculate protection leg (PV of expected losses)
    protection_leg = expected_tranche_loss[i] * notionals.sum() * np.exp(-discount_rate * maturity)
    
    # Assume premiums paid annually until maturity or default
    premium_leg = 0
    for t in range(1, maturity + 1):
        survival_prob = 1 - loss_probability(rho,attachment,detachment,t,default_times)  # Simplified survival
        premium_leg += notionals.sum() * survival_prob * np.exp(-discount_rate * t)
    
    # Solve for spread such that premium_leg = protection_leg
    spread[i] = protection_leg / premium_leg
    i += 1
    
fig,axes = plt.subplots(2,1,figsize=(8, 8))
axes[0].plot(np.arange(n_tranches),expected_tranche_loss)
axes[0].set_title('Expected tranches losses')
axes[0].set_xlabel('Tranch')
axes[0].set_ylabel('Losses')
axes[0].set_xticks([0,1,2,3,4],tranche_name,rotation=0)
axes[0].grid(True)

axes[1].plot(np.arange(n_tranches),100*spread)
axes[1].set_title('Spread')
axes[1].set_xlabel('Tranch')
axes[1].set_ylabel('%')
axes[1].set_xticks([0,1,2,3,4],tranche_name,rotation=0)
axes[1].grid(True)

plt.tight_layout()
plt.show()    