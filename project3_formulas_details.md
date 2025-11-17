# Project 3: Financial Portfolio Analyzer - Formulas & Detailed Concepts

## 📐 Mathematical Formulas for Portfolio Analysis

### 1. Return Calculations

#### Simple Return
```
R = (P₁ - P₀) / P₀
```
Where:
- R = Simple return
- P₁ = Ending price
- P₀ = Beginning price
**NumPy Implementation:**
```python
simple_returns = (prices[1:] - prices[:-1]) / prices[:-1]
```

#### Log Return (Continuously Compounded)
```
r = ln(P₁ / P₀) = ln(P₁) - ln(P₀)
```
**Advantages:** Time-additive, symmetric
**NumPy Implementation:**
```python
log_returns = np.log(prices[1:] / prices[:-1])
# or
log_returns = np.diff(np.log(prices))
```

#### Portfolio Return
```
Rp = Σ(wi × Ri)
```
Where:
- Rp = Portfolio return
- wi = Weight of asset i
- Ri = Return of asset i
**NumPy Implementation:**
```python
portfolio_return = np.dot(weights, asset_returns)
```

### 2. Risk Measures

#### Portfolio Variance
```
σp² = Σ Σ wi × wj × σij
```
Matrix form:
```
σp² = wᵀ × Σ × w
```
Where:
- w = Weight vector
- Σ = Covariance matrix
- σij = Covariance between assets i and j
**NumPy Implementation:**
```python
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
portfolio_volatility = np.sqrt(portfolio_variance)
```

#### Covariance Matrix
```
Σij = E[(Ri - μi)(Rj - μj)]
```
**NumPy Implementation:**
```python
# Method 1: Using np.cov
cov_matrix = np.cov(returns.T)  # Transpose for correct orientation

# Method 2: Manual calculation
mean_returns = np.mean(returns, axis=0)
centered_returns = returns - mean_returns
cov_matrix = np.dot(centered_returns.T, centered_returns) / (len(returns) - 1)
```

#### Correlation Matrix
```
ρij = σij / (σi × σj)
```
**NumPy Implementation:**
```python
corr_matrix = np.corrcoef(returns.T)
# or from covariance matrix
std_devs = np.sqrt(np.diag(cov_matrix))
corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
```

### 3. Risk-Adjusted Performance Metrics

#### Sharpe Ratio
```
SR = (Rp - Rf) / σp
```
Where:
- SR = Sharpe ratio
- Rp = Portfolio return
- Rf = Risk-free rate
- σp = Portfolio standard deviation
**Interpretation:** Higher is better (more return per unit of risk)
**NumPy Implementation:**
```python
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
```

#### Information Ratio
```
IR = (Rp - Rb) / TE
```
Where:
- IR = Information ratio
- Rp = Portfolio return
- Rb = Benchmark return
- TE = Tracking error = σ(Rp - Rb)
**NumPy Implementation:**
```python
active_returns = portfolio_returns - benchmark_returns
tracking_error = np.std(active_returns)
information_ratio = np.mean(active_returns) / tracking_error
```

#### Sortino Ratio
```
Sortino = (Rp - Rf) / DD
```
Where:
- DD = Downside deviation (volatility of negative returns only)
**NumPy Implementation:**
```python
negative_returns = portfolio_returns[portfolio_returns < 0]
downside_deviation = np.std(negative_returns)
sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation
```

### 4. Value at Risk (VaR)

#### Parametric VaR (Normal Distribution)
```
VaR = μ - z × σ
```
Where:
- μ = Expected return
- z = Z-score for confidence level
- σ = Standard deviation
**For 95% confidence:** z = 1.645
**For 99% confidence:** z = 2.326
**NumPy Implementation:**
```python
confidence_level = 0.95
z_score = np.percentile(np.random.normal(0, 1, 10000), (1-confidence_level)*100)
var_parametric = portfolio_return + z_score * portfolio_volatility
```

#### Historical VaR
```
VaR = Percentile of historical returns
```
**NumPy Implementation:**
```python
var_historical = np.percentile(portfolio_returns, (1-confidence_level)*100)
```

#### Monte Carlo VaR
```python
# Generate random scenarios
n_simulations = 10000
random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_simulations)
portfolio_scenarios = np.dot(random_returns, weights)
var_monte_carlo = np.percentile(portfolio_scenarios, (1-confidence_level)*100)
```

### 5. Beta and Market Risk

#### Beta Coefficient
```
βi = Cov(Ri, Rm) / Var(Rm)
```
Where:
- βi = Beta of asset i
- Ri = Return of asset i
- Rm = Market return
**Interpretation:**
- β > 1: More volatile than market
- β < 1: Less volatile than market
- β = 1: Same volatility as market
**NumPy Implementation:**
```python
# Method 1: Using covariance
beta = np.cov(asset_returns, market_returns)[0, 1] / np.var(market_returns)

# Method 2: Using linear regression
beta = np.polyfit(market_returns, asset_returns, 1)[0]
```

#### Portfolio Beta
```
βp = Σ(wi × βi)
```
**NumPy Implementation:**
```python
portfolio_beta = np.dot(weights, individual_betas)
```

### 6. Capital Asset Pricing Model (CAPM)

#### Expected Return
```
E(Ri) = Rf + βi × [E(Rm) - Rf]
```
Where:
- E(Ri) = Expected return of asset i
- Rf = Risk-free rate
- βi = Beta of asset i
- E(Rm) = Expected market return
- [E(Rm) - Rf] = Market risk premium
**NumPy Implementation:**
```python
expected_returns = risk_free_rate + betas * (market_return - risk_free_rate)
```

#### Security Market Line (SML)
```python
# Plot expected return vs beta
betas = np.linspace(0, 2, 100)
sml_returns = risk_free_rate + betas * market_risk_premium
```

### 7. Portfolio Optimization

#### Minimum Variance Portfolio
```
Minimize: wᵀ × Σ × w
Subject to: Σwi = 1
```
**Analytical Solution:**
```
w = (Σ⁻¹ × 1) / (1ᵀ × Σ⁻¹ × 1)
```
**NumPy Implementation:**
```python
inv_cov = np.linalg.inv(cov_matrix)
ones = np.ones((len(mean_returns), 1))
w_mvp = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
```

#### Maximum Sharpe Ratio Portfolio
```
Maximize: (wᵀμ - Rf) / √(wᵀΣw)
```
**Analytical Solution:**
```
w = Σ⁻¹ × (μ - Rf × 1) / (1ᵀ × Σ⁻¹ × (μ - Rf × 1))
```
**NumPy Implementation:**
```python
excess_returns = mean_returns - risk_free_rate
inv_cov = np.linalg.inv(cov_matrix)
ones = np.ones(len(mean_returns))
numerator = np.dot(inv_cov, excess_returns)
denominator = np.dot(ones, numerator)
w_tangent = numerator / denominator
```

### 8. Efficient Frontier

#### Two-Asset Case
```
σp² = w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁₂
μp = w₁μ₁ + w₂μ₂
```
Where w₂ = 1 - w₁
**NumPy Implementation:**
```python
weights_1 = np.linspace(0, 1, 100)
weights_2 = 1 - weights_1
portfolio_returns = weights_1 * return_1 + weights_2 * return_2
portfolio_variance = (weights_1**2 * var_1 + weights_2**2 * var_2 + 
                     2 * weights_1 * weights_2 * cov_12)
portfolio_volatility = np.sqrt(portfolio_variance)
```

#### Multi-Asset Efficient Frontier
```python
def efficient_frontier(mean_returns, cov_matrix, target_returns):
    n_assets = len(mean_returns)
    results = []
    
    for target in target_returns:
        # Quadratic programming problem
        # Minimize: 0.5 * w.T @ cov_matrix @ w
        # Subject to: mean_returns.T @ w = target, sum(w) = 1
        
        # Using Lagrange multipliers (analytical solution)
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones((n_assets, 1))
        
        A = np.dot(mean_returns.T, np.dot(inv_cov, mean_returns))
        B = np.dot(mean_returns.T, np.dot(inv_cov, ones))
        C = np.dot(ones.T, np.dot(inv_cov, ones))
        
        lambda_1 = (C * target - B) / (A * C - B**2)
        lambda_2 = (B * target - A) / (B**2 - A * C)
        
        weights = lambda_1 * np.dot(inv_cov, mean_returns) + lambda_2 * np.dot(inv_cov, ones)
        variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        
        results.append({
            'target_return': target,
            'volatility': np.sqrt(variance),
            'weights': weights.flatten()
        })
    
    return results
```

### 9. Risk Decomposition

#### Marginal Risk Contribution
```
MRCi = ∂σp/∂wi = (Σw)i / σp
```
**NumPy Implementation:**
```python
marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
```

#### Component Risk Contribution
```
CRCi = wi × MRCi
```
**NumPy Implementation:**
```python
component_contrib = weights * marginal_contrib
# Verify: sum(component_contrib) should equal portfolio_volatility
```

#### Percentage Risk Contribution
```
PRCi = CRCi / σp × 100%
```
**NumPy Implementation:**
```python
percent_contrib = component_contrib / portfolio_volatility * 100
```

### 10. Performance Attribution

#### Brinson Model (Asset Allocation vs Security Selection)
```
Total Return = Allocation Effect + Selection Effect + Interaction Effect
```

**Allocation Effect:**
```
AA = Σ(wpi - wbi) × rbi
```

**Selection Effect:**
```
SS = Σwbi × (rpi - rbi)
```

**Interaction Effect:**
```
IN = Σ(wpi - wbi) × (rpi - rbi)
```
Where:
- wpi, wbi = Portfolio and benchmark weights
- rpi, rbi = Portfolio and benchmark returns
**NumPy Implementation:**
```python
weight_diff = portfolio_weights - benchmark_weights
return_diff = portfolio_returns - benchmark_returns

allocation_effect = np.sum(weight_diff * benchmark_returns)
selection_effect = np.sum(benchmark_weights * return_diff)
interaction_effect = np.sum(weight_diff * return_diff)
```

## 🧮 Advanced NumPy Concepts

### 1. Linear Algebra Operations

#### Matrix Decompositions
```python
# Eigenvalue decomposition
eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

# Singular Value Decomposition
U, s, Vt = np.linalg.svd(returns_matrix)

# Cholesky decomposition (for positive definite matrices)
L = np.linalg.cholesky(cov_matrix)
```

#### Matrix Operations
```python
# Matrix inversion
inv_cov = np.linalg.inv(cov_matrix)

# Pseudo-inverse (for singular matrices)
pinv_cov = np.linalg.pinv(cov_matrix)

# Matrix rank
rank = np.linalg.matrix_rank(cov_matrix)

# Condition number
cond_num = np.linalg.cond(cov_matrix)
```

### 2. Statistical Operations

#### Moments and Distributions
```python
# Central moments
mean = np.mean(returns, axis=0)
variance = np.var(returns, axis=0)
skewness = scipy.stats.skew(returns, axis=0)
kurtosis = scipy.stats.kurtosis(returns, axis=0)

# Rolling statistics
def rolling_stats(data, window):
    rolling_mean = np.convolve(data, np.ones(window)/window, mode='valid')
    rolling_var = np.array([np.var(data[i:i+window]) for i in range(len(data)-window+1)])
    return rolling_mean, rolling_var
```

#### Hypothesis Testing
```python
# T-test for mean return significance
from scipy import stats
t_stat, p_value = stats.ttest_1samp(returns, 0)

# Jarque-Bera test for normality
jb_stat, jb_pvalue = stats.jarque_bera(returns)
```

### 3. Monte Carlo Methods

#### Correlated Random Variables
```python
# Generate correlated returns using Cholesky decomposition
def generate_correlated_returns(mean_returns, cov_matrix, n_simulations):
    n_assets = len(mean_returns)
    
    # Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)
    
    # Generate independent random variables
    Z = np.random.normal(0, 1, (n_simulations, n_assets))
    
    # Transform to correlated variables
    correlated_returns = mean_returns + np.dot(Z, L.T)
    
    return correlated_returns
```

#### Bootstrap Methods
```python
def bootstrap_portfolio_metrics(returns, weights, n_bootstrap=1000):
    n_observations = len(returns)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_observations, n_observations, replace=True)
        bootstrap_returns = returns[bootstrap_indices]
        
        # Calculate portfolio metrics
        portfolio_returns = np.dot(bootstrap_returns, weights)
        metrics = {
            'mean': np.mean(portfolio_returns),
            'volatility': np.std(portfolio_returns),
            'sharpe': np.mean(portfolio_returns) / np.std(portfolio_returns)
        }
        bootstrap_metrics.append(metrics)
    
    return bootstrap_metrics
```

### 4. Time Series Analysis

#### Autocorrelation
```python
def autocorrelation(data, max_lags=20):
    n = len(data)
    data_centered = data - np.mean(data)
    autocorr = np.correlate(data_centered, data_centered, mode='full')
    autocorr = autocorr[n-1:n-1+max_lags+1]
    autocorr = autocorr / autocorr[0]  # Normalize
    return autocorr
```

#### Moving Averages and Trends
```python
# Simple moving average
def simple_moving_average(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

# Exponential moving average
def exponential_moving_average(prices, alpha):
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema
```

### 5. Optimization Techniques

#### Gradient Descent for Portfolio Optimization
```python
def gradient_descent_portfolio(mean_returns, cov_matrix, learning_rate=0.01, max_iter=1000):
    n_assets = len(mean_returns)
    weights = np.ones(n_assets) / n_assets  # Equal weights initialization
    
    for i in range(max_iter):
        # Calculate gradient of negative Sharpe ratio
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Gradient calculation
        grad_return = mean_returns
        grad_volatility = np.dot(cov_matrix, weights) / portfolio_volatility
        
        # Negative Sharpe ratio gradient
        sharpe_gradient = -(grad_return * portfolio_volatility - portfolio_return * grad_volatility) / (portfolio_volatility**2)
        
        # Update weights
        weights -= learning_rate * sharpe_gradient
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Ensure non-negative weights (long-only constraint)
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
    
    return weights
```

## 📊 Performance Metrics Implementation

### Risk-Adjusted Returns
```python
def calculate_performance_metrics(returns, benchmark_returns=None, risk_free_rate=0.02):
    metrics = {}
    
    # Basic statistics
    metrics['mean_return'] = np.mean(returns)
    metrics['volatility'] = np.std(returns)
    metrics['skewness'] = scipy.stats.skew(returns)
    metrics['kurtosis'] = scipy.stats.kurtosis(returns)
    
    # Risk-adjusted metrics
    metrics['sharpe_ratio'] = (metrics['mean_return'] - risk_free_rate) / metrics['volatility']
    
    # Downside metrics
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        metrics['downside_deviation'] = np.std(negative_returns)
        metrics['sortino_ratio'] = (metrics['mean_return'] - risk_free_rate) / metrics['downside_deviation']
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    metrics['max_drawdown'] = np.min(drawdown)
    
    # Calmar ratio
    if metrics['max_drawdown'] != 0:
        metrics['calmar_ratio'] = metrics['mean_return'] / abs(metrics['max_drawdown'])
    
    # Benchmark-relative metrics
    if benchmark_returns is not None:
        active_returns = returns - benchmark_returns
        metrics['tracking_error'] = np.std(active_returns)
        metrics['information_ratio'] = np.mean(active_returns) / metrics['tracking_error']
        
        # Beta calculation
        metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        
        # Alpha calculation (Jensen's alpha)
        metrics['alpha'] = metrics['mean_return'] - (risk_free_rate + metrics['beta'] * (np.mean(benchmark_returns) - risk_free_rate))
    
    return metrics
```

Ye comprehensive mathematical foundation aapko financial portfolio analysis ke saare concepts aur NumPy implementations ki deep understanding dega!