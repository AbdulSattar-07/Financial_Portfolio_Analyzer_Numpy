# Project 3: Financial Portfolio Analyzer - Complete Guide

## 💰 Project Overview
Build a comprehensive financial portfolio analysis system using NumPy for advanced mathematical operations. This project teaches linear algebra, statistical analysis, and financial modeling through practical investment portfolio management.

## 🎯 Learning Objectives
By completing this project, you will master:
- Matrix operations and linear algebra with NumPy
- Advanced statistical analysis and risk calculations
- Random number generation and Monte Carlo simulations
- Correlation and covariance matrix computations
- Portfolio optimization using mathematical models
- Time series analysis and financial metrics
- Vectorized operations for performance optimization

## 📊 Mathematical Formulas Used

### 1. Portfolio Return Calculation
```
Portfolio Return = Σ(wi × ri)
```
Where:
- wi = Weight of asset i in portfolio
- ri = Return of asset i
**NumPy:** `np.dot(weights, returns)`

### 2. Portfolio Variance (Risk)
```
σp² = wᵀ × Σ × w
```
Where:
- w = Weight vector
- Σ = Covariance matrix
- wᵀ = Transpose of weight vector
**NumPy:** `np.dot(weights.T, np.dot(cov_matrix, weights))`

### 3. Sharpe Ratio
```
Sharpe Ratio = (Rp - Rf) / σp
```
Where:
- Rp = Portfolio return
- Rf = Risk-free rate
- σp = Portfolio standard deviation
**Purpose:** Risk-adjusted return measurement

### 4. Beta Coefficient
```
βi = Cov(ri, rm) / Var(rm)
```
Where:
- ri = Asset return
- rm = Market return
**NumPy:** `np.cov(asset_returns, market_returns)[0,1] / np.var(market_returns)`

### 5. Value at Risk (VaR)
```
VaR = μ - z × σ
```
Where:
- μ = Expected return
- z = Z-score for confidence level
- σ = Standard deviation
**Purpose:** Maximum expected loss at given confidence level

### 6. Correlation Matrix
```
ρij = Cov(ri, rj) / (σi × σj)
```
**NumPy:** `np.corrcoef(returns_matrix)`

### 7. Expected Return (CAPM)
```
E(ri) = Rf + βi × (E(rm) - Rf)
```
Where:
- E(ri) = Expected return of asset i
- Rf = Risk-free rate
- βi = Beta of asset i
- E(rm) = Expected market return

### 8. Compound Annual Growth Rate (CAGR)
```
CAGR = (Ending Value / Beginning Value)^(1/n) - 1
```
Where n = number of years

### 9. Maximum Drawdown
```
MDD = (Peak Value - Trough Value) / Peak Value
```
**Purpose:** Largest peak-to-trough decline

### 10. Information Ratio
```
IR = (Rp - Rb) / σ(Rp - Rb)
```
Where:
- Rp = Portfolio return
- Rb = Benchmark return
- σ(Rp - Rb) = Tracking error

## 🔄 Complete Project Steps

### Step 1: Data Generation and Market Simulation
```python
# Generate realistic stock price data
np.random.seed(42)
n_assets = 5
n_days = 252  # Trading days in a year

# Simulate correlated asset returns
correlation_matrix = create_correlation_matrix(n_assets)
returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
```

### Step 2: Portfolio Construction
```python
# Define portfolio weights
weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # Must sum to 1.0

# Calculate portfolio metrics
portfolio_return = np.dot(weights, mean_returns)
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
portfolio_volatility = np.sqrt(portfolio_variance)
```

### Step 3: Risk Analysis
```python
# Calculate Value at Risk (95% confidence)
confidence_level = 0.95
z_score = np.percentile(np.random.normal(0, 1, 10000), (1-confidence_level)*100)
var_95 = portfolio_return + z_score * portfolio_volatility

# Calculate Maximum Drawdown
cumulative_returns = np.cumprod(1 + daily_returns)
running_max = np.maximum.accumulate(cumulative_returns)
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = np.min(drawdown)
```

### Step 4: Performance Metrics
```python
# Sharpe Ratio calculation
risk_free_rate = 0.02  # 2% annual
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

# Beta calculation (vs market)
market_returns = generate_market_data()
portfolio_beta = np.cov(portfolio_returns, market_returns)[0,1] / np.var(market_returns)
```

### Step 5: Correlation Analysis
```python
# Calculate correlation matrix
correlation_matrix = np.corrcoef(asset_returns.T)

# Find highly correlated pairs
high_corr_pairs = np.where(np.abs(correlation_matrix) > 0.7)
```

### Step 6: Monte Carlo Simulation
```python
# Monte Carlo portfolio simulation
n_simulations = 10000
simulated_returns = []

for _ in range(n_simulations):
    # Generate random returns
    random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
    portfolio_return = np.dot(weights, random_returns)
    simulated_returns.append(portfolio_return)

# Analyze simulation results
var_monte_carlo = np.percentile(simulated_returns, 5)  # 5% VaR
```

### Step 7: Portfolio Optimization
```python
# Efficient Frontier calculation
def calculate_efficient_frontier(mean_returns, cov_matrix, target_returns):
    efficient_portfolios = []
    
    for target in target_returns:
        # Minimize risk for given return
        optimal_weights = optimize_portfolio(mean_returns, cov_matrix, target)
        efficient_portfolios.append(optimal_weights)
    
    return efficient_portfolios
```

### Step 8: Backtesting
```python
# Historical performance analysis
def backtest_portfolio(weights, historical_returns):
    portfolio_returns = np.dot(historical_returns, weights)
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    
    metrics = {
        'total_return': cumulative_returns[-1] - 1,
        'volatility': np.std(portfolio_returns) * np.sqrt(252),
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_returns),
        'max_drawdown': calculate_max_drawdown(cumulative_returns)
    }
    
    return metrics
```

### Step 9: Risk Attribution
```python
# Component risk analysis
def calculate_risk_contribution(weights, cov_matrix):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    marginal_contrib = np.dot(cov_matrix, weights)
    contrib = weights * marginal_contrib / portfolio_variance
    return contrib
```

### Step 10: Advanced Analytics
```python
# Rolling statistics
def calculate_rolling_metrics(returns, window=30):
    rolling_mean = np.convolve(returns, np.ones(window)/window, mode='valid')
    rolling_std = np.array([np.std(returns[i:i+window]) for i in range(len(returns)-window+1)])
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    
    return rolling_mean, rolling_std, rolling_sharpe
```

## 🧮 NumPy Concepts Covered

### 1. Linear Algebra Operations
```python
# Matrix multiplication
portfolio_return = np.dot(weights, returns)
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

# Matrix inversion
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
```

### 2. Statistical Functions
```python
# Covariance and correlation
cov_matrix = np.cov(returns.T)
corr_matrix = np.corrcoef(returns.T)

# Percentiles for VaR
var_95 = np.percentile(returns, 5)
var_99 = np.percentile(returns, 1)

# Statistical moments
mean_return = np.mean(returns, axis=0)
std_return = np.std(returns, axis=0)
skewness = scipy.stats.skew(returns, axis=0)
kurtosis = scipy.stats.kurtosis(returns, axis=0)
```

### 3. Random Number Generation
```python
# Multivariate normal distribution
returns = np.random.multivariate_normal(mean_vector, cov_matrix, n_samples)

# Monte Carlo simulations
np.random.seed(42)  # Reproducible results
simulations = np.random.normal(mu, sigma, (n_sims, n_days))

# Bootstrap sampling
bootstrap_samples = np.random.choice(returns, size=(n_bootstrap, len(returns)), replace=True)
```

### 4. Array Broadcasting and Vectorization
```python
# Vectorized return calculations
daily_returns = (prices[1:] - prices[:-1]) / prices[:-1]

# Broadcasting for portfolio calculations
weighted_returns = returns * weights[:, np.newaxis]  # Broadcasting weights

# Cumulative products for growth
cumulative_wealth = np.cumprod(1 + returns, axis=0)
```

### 5. Advanced Indexing
```python
# Boolean indexing for filtering
positive_returns = returns[returns > 0]
bear_market_days = returns[returns < -0.02]

# Fancy indexing for rebalancing
rebalance_dates = np.arange(0, len(returns), 21)  # Monthly rebalancing
rebalanced_returns = returns[rebalance_dates]
```

### 6. Time Series Operations
```python
# Rolling window calculations
def rolling_window(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

# Moving averages
ma_20 = np.convolve(prices, np.ones(20)/20, mode='valid')
ma_50 = np.convolve(prices, np.ones(50)/50, mode='valid')
```

## 📈 Financial Concepts Explained

### 1. Modern Portfolio Theory (MPT)
- **Diversification Benefits:** Reduce risk through uncorrelated assets
- **Efficient Frontier:** Optimal risk-return combinations
- **Capital Allocation Line:** Risk-free asset + risky portfolio

### 2. Risk Measures
- **Standard Deviation:** Volatility measure
- **Value at Risk (VaR):** Potential loss at confidence level
- **Conditional VaR:** Expected loss beyond VaR
- **Maximum Drawdown:** Worst peak-to-trough decline

### 3. Performance Metrics
- **Sharpe Ratio:** Risk-adjusted return
- **Information Ratio:** Active return vs tracking error
- **Sortino Ratio:** Downside risk-adjusted return
- **Calmar Ratio:** Return vs maximum drawdown

### 4. Asset Pricing Models
- **CAPM:** Capital Asset Pricing Model
- **Fama-French:** Multi-factor model
- **APT:** Arbitrage Pricing Theory

## 🔍 Practical Applications

### Investment Management
- Portfolio construction and optimization
- Risk budgeting and allocation
- Performance attribution analysis
- Benchmark comparison

### Risk Management
- Value at Risk calculations
- Stress testing scenarios
- Correlation monitoring
- Drawdown analysis

### Quantitative Finance
- Factor model development
- Alpha generation strategies
- Market neutral portfolios
- Statistical arbitrage

### Regulatory Compliance
- Capital adequacy calculations
- Risk reporting requirements
- Stress test scenarios
- Model validation

## 💡 Key Learning Points

1. **Linear algebra is fundamental** to portfolio mathematics
2. **Correlation analysis reveals** diversification opportunities
3. **Monte Carlo simulation** provides robust risk estimates
4. **Vectorization dramatically improves** calculation performance
5. **Matrix operations enable** efficient portfolio optimization
6. **Statistical analysis** drives investment decisions
7. **Risk management** is as important as return generation

## 🛠️ Tools and Libraries

### Core Libraries
- **NumPy** - Mathematical operations and linear algebra
- **SciPy** - Advanced statistical functions
- **Pandas** - Data manipulation (optional)
- **Matplotlib/Plotly** - Visualization

### Financial Extensions
- **QuantLib** - Quantitative finance library
- **PyPortfolioOpt** - Portfolio optimization
- **Zipline** - Backtesting framework
- **Alpha Architect** - Factor analysis

## 📝 Project Structure
```
project3_portfolio_analyzer/
├── project3_portfolio_analyzer.py       # Main implementation
├── project3_formulas_details.md         # Mathematical formulas
├── project3_README.md                   # Step-by-step guide
├── sample_data/                         # Historical data
├── results/                             # Analysis outputs
└── portfolio_analyzer_streamlit.py      # Interactive UI
```

## 📊 Expected Outputs

### Portfolio Metrics
- Expected return and volatility
- Sharpe ratio and other ratios
- Beta and correlation analysis
- Risk contribution by asset

### Visualizations
- Efficient frontier plot
- Correlation heatmap
- Performance charts
- Risk decomposition

### Risk Analysis
- Value at Risk calculations
- Monte Carlo simulation results
- Stress test scenarios
- Maximum drawdown analysis

This comprehensive guide provides everything needed to master financial portfolio analysis with NumPy through practical quantitative finance applications!