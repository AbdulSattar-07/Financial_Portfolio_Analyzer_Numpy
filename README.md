# Project 3: Financial Portfolio Analyzer - Step-by-Step Guide

## 💰 Overview
This project teaches advanced NumPy concepts through practical financial portfolio analysis. You'll learn linear algebra, statistical analysis, and quantitative finance techniques while building a comprehensive portfolio management system.

## 🎯 Learning Objectives
- Master matrix operations and linear algebra with NumPy
- Understand advanced statistical analysis for finance
- Learn Monte Carlo simulation techniques
- Practice correlation and covariance analysis
- Implement portfolio optimization algorithms
- Calculate financial risk metrics and performance indicators

## 📁 Project Structure
```
project3_portfolio_analyzer/
├── project3_portfolio_analyzer.py       # Main implementation
├── project3_formulas_details.md         # Mathematical formulas
├── project3_README.md                   # This guide
├── sample_data/                         # Historical market data
├── results/                             # Analysis outputs
├── portfolio_analyzer_streamlit.py      # Interactive UI
└── requirements.txt                     # Dependencies
```

## 🔧 Prerequisites
```bash
pip install numpy scipy matplotlib plotly pandas yfinance
```

## 📊 Key NumPy Functions You'll Learn

| Function | Purpose | Financial Application |
|----------|---------|----------------------|
| `np.dot()` | Matrix multiplication | Portfolio return calculation |
| `np.linalg.inv()` | Matrix inversion | Portfolio optimization |
| `np.cov()` | Covariance matrix | Risk analysis |
| `np.corrcoef()` | Correlation matrix | Diversification analysis |
| `np.random.multivariate_normal()` | Correlated random variables | Monte Carlo simulation |
| `np.percentile()` | Percentile calculation | Value at Risk (VaR) |
| `np.linalg.eig()` | Eigenvalue decomposition | Principal component analysis |
| `np.cumsum()` | Cumulative sum | Cumulative returns |

## 🚀 Step-by-Step Implementation

### Step 1: Market Data Generation and Loading
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    def __init__(self, n_assets=5, n_days=252):
        self.n_assets = n_assets
        self.n_days = n_days
        self.asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
        
    def generate_market_data(self):
        """Generate realistic market data with correlations"""
        np.random.seed(42)
        
        # Define expected returns (annual)
        expected_returns = np.array([0.08, 0.12, 0.10, 0.15, 0.06])
        
        # Create correlation matrix
        correlation_matrix = self.create_correlation_matrix()
        
        # Convert to daily returns
        daily_returns = expected_returns / 252
        
        # Create covariance matrix
        volatilities = np.array([0.15, 0.25, 0.20, 0.30, 0.10])  # Annual volatilities
        daily_volatilities = volatilities / np.sqrt(252)
        
        # Covariance = correlation * vol_i * vol_j
        cov_matrix = np.outer(daily_volatilities, daily_volatilities) * correlation_matrix
        
        # Generate correlated returns
        returns = np.random.multivariate_normal(daily_returns, cov_matrix, self.n_days)
        
        return returns, daily_returns, cov_matrix, correlation_matrix
```

**NumPy Concepts:**
- `np.random.multivariate_normal()` for correlated random variables
- `np.outer()` for creating covariance matrix
- Matrix operations for financial modeling

### Step 2: Portfolio Construction
```python
def create_portfolio(self, weights=None):
    """Create portfolio with given weights"""
    if weights is None:
        # Equal weight portfolio
        weights = np.ones(self.n_assets) / self.n_assets
    
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    # Calculate portfolio metrics
    portfolio_return = np.dot(weights, self.expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    return {
        'weights': weights,
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
        'variance': portfolio_variance
    }
```

**NumPy Concepts:**
- `np.dot()` for matrix multiplication
- Vector-matrix operations
- Portfolio mathematics implementation

### Step 3: Risk Analysis
```python
def calculate_var(self, portfolio_returns, confidence_level=0.95):
    """Calculate Value at Risk using multiple methods"""
    
    # Historical VaR
    var_historical = np.percentile(portfolio_returns, (1-confidence_level)*100)
    
    # Parametric VaR (assuming normal distribution)
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    z_score = stats.norm.ppf(1-confidence_level)
    var_parametric = mean_return + z_score * std_return
    
    # Monte Carlo VaR
    n_simulations = 10000
    simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
    var_monte_carlo = np.percentile(simulated_returns, (1-confidence_level)*100)
    
    return {
        'historical': var_historical,
        'parametric': var_parametric,
        'monte_carlo': var_monte_carlo
    }

def calculate_max_drawdown(self, returns):
    """Calculate maximum drawdown"""
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    return max_drawdown, drawdown
```

**NumPy Concepts:**
- `np.percentile()` for VaR calculation
- `np.cumprod()` for cumulative returns
- `np.maximum.accumulate()` for running maximum

### Step 4: Performance Metrics
```python
def calculate_performance_metrics(self, returns, benchmark_returns=None, risk_free_rate=0.02):
    """Calculate comprehensive performance metrics"""
    
    # Convert to daily risk-free rate
    daily_rf = risk_free_rate / 252
    
    # Basic statistics
    mean_return = np.mean(returns)
    volatility = np.std(returns) * np.sqrt(252)  # Annualized
    
    # Sharpe Ratio
    excess_returns = returns - daily_rf
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    # Sortino Ratio (downside risk only)
    negative_returns = returns[returns < daily_rf]
    if len(negative_returns) > 0:
        downside_deviation = np.std(negative_returns) * np.sqrt(252)
        sortino_ratio = (mean_return * 252 - risk_free_rate) / downside_deviation
    else:
        sortino_ratio = np.inf
    
    # Maximum Drawdown
    max_drawdown, _ = self.calculate_max_drawdown(returns)
    
    # Calmar Ratio
    calmar_ratio = (mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    metrics = {
        'annual_return': mean_return * 252,
        'annual_volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }
    
    # Benchmark-relative metrics
    if benchmark_returns is not None:
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
        
        # Beta calculation
        beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        
        # Alpha (Jensen's alpha)
        benchmark_annual = np.mean(benchmark_returns) * 252
        alpha = metrics['annual_return'] - (risk_free_rate + beta * (benchmark_annual - risk_free_rate))
        
        metrics.update({
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha
        })
    
    return metrics
```

**NumPy Concepts:**
- Statistical functions: `np.mean()`, `np.std()`, `np.var()`
- Boolean indexing for downside risk
- `np.cov()` for beta calculation

### Step 5: Correlation Analysis
```python
def analyze_correlations(self, returns):
    """Comprehensive correlation analysis"""
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(returns.T)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(self.n_assets):
        for j in range(i+1, self.n_assets):
            corr_value = corr_matrix[i, j]
            if abs(corr_value) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    'asset1': self.asset_names[i],
                    'asset2': self.asset_names[j],
                    'correlation': corr_value
                })
    
    # Calculate average correlations
    upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    avg_correlation = np.mean(upper_triangle)
    
    # Diversification ratio
    portfolio_weights = np.ones(self.n_assets) / self.n_assets
    weighted_avg_vol = np.dot(portfolio_weights, np.sqrt(np.diag(self.cov_matrix)) * np.sqrt(252))
    portfolio_vol = np.sqrt(np.dot(portfolio_weights.T, np.dot(self.cov_matrix, portfolio_weights))) * np.sqrt(252)
    diversification_ratio = weighted_avg_vol / portfolio_vol
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_corr_pairs,
        'average_correlation': avg_correlation,
        'diversification_ratio': diversification_ratio
    }
```

**NumPy Concepts:**
- `np.corrcoef()` for correlation matrix
- `np.triu_indices_from()` for upper triangle extraction
- Matrix indexing and boolean operations

### Step 6: Monte Carlo Simulation
```python
def monte_carlo_simulation(self, n_simulations=10000, time_horizon=252):
    """Run Monte Carlo simulation for portfolio scenarios"""
    
    # Portfolio weights (equal weight for this example)
    weights = np.ones(self.n_assets) / self.n_assets
    
    # Generate random scenarios
    scenarios = np.random.multivariate_normal(
        self.expected_returns, 
        self.cov_matrix, 
        (n_simulations, time_horizon)
    )
    
    # Calculate portfolio returns for each scenario
    portfolio_scenarios = np.dot(scenarios, weights)
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_scenarios, axis=1)
    final_values = cumulative_returns[:, -1]
    
    # Calculate statistics
    simulation_stats = {
        'mean_final_value': np.mean(final_values),
        'median_final_value': np.median(final_values),
        'std_final_value': np.std(final_values),
        'percentile_5': np.percentile(final_values, 5),
        'percentile_95': np.percentile(final_values, 95),
        'probability_loss': np.mean(final_values < 1.0),
        'expected_shortfall': np.mean(final_values[final_values < np.percentile(final_values, 5)])
    }
    
    return simulation_stats, final_values, cumulative_returns
```

**NumPy Concepts:**
- `np.random.multivariate_normal()` for correlated scenarios
- `np.cumprod()` for cumulative returns
- Advanced statistical analysis with percentiles

### Step 7: Portfolio Optimization
```python
def optimize_portfolio(self, target_return=None, method='min_variance'):
    """Portfolio optimization using analytical solutions"""
    
    # Minimum variance portfolio
    if method == 'min_variance':
        inv_cov = np.linalg.inv(self.cov_matrix)
        ones = np.ones((self.n_assets, 1))
        
        # Analytical solution: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
        numerator = np.dot(inv_cov, ones)
        denominator = np.dot(ones.T, numerator)
        optimal_weights = (numerator / denominator).flatten()
    
    # Maximum Sharpe ratio portfolio
    elif method == 'max_sharpe':
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = self.expected_returns - risk_free_rate
        
        inv_cov = np.linalg.inv(self.cov_matrix)
        ones = np.ones(self.n_assets)
        
        # Analytical solution for tangency portfolio
        numerator = np.dot(inv_cov, excess_returns)
        denominator = np.dot(ones, numerator)
        optimal_weights = numerator / denominator
    
    # Target return portfolio
    elif method == 'target_return' and target_return is not None:
        inv_cov = np.linalg.inv(self.cov_matrix)
        ones = np.ones((self.n_assets, 1))
        mu = self.expected_returns.reshape(-1, 1)
        
        # Lagrange multiplier method
        A = np.dot(mu.T, np.dot(inv_cov, mu))[0, 0]
        B = np.dot(mu.T, np.dot(inv_cov, ones))[0, 0]
        C = np.dot(ones.T, np.dot(inv_cov, ones))[0, 0]
        
        lambda_1 = (C * target_return - B) / (A * C - B**2)
        lambda_2 = (B * target_return - A) / (B**2 - A * C)
        
        optimal_weights = (lambda_1 * np.dot(inv_cov, mu) + lambda_2 * np.dot(inv_cov, ones)).flatten()
    
    else:
        raise ValueError("Invalid optimization method or missing target return")
    
    # Ensure weights are valid
    optimal_weights = np.maximum(optimal_weights, 0)  # Long-only constraint
    optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize
    
    return optimal_weights
```

**NumPy Concepts:**
- `np.linalg.inv()` for matrix inversion
- Linear algebra for portfolio optimization
- Lagrange multiplier implementation

### Step 8: Efficient Frontier
```python
def calculate_efficient_frontier(self, n_points=50):
    """Calculate the efficient frontier"""
    
    # Define range of target returns
    min_return = np.min(self.expected_returns)
    max_return = np.max(self.expected_returns)
    target_returns = np.linspace(min_return, max_return, n_points)
    
    efficient_portfolios = []
    
    for target in target_returns:
        try:
            weights = self.optimize_portfolio(target_return=target, method='target_return')
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualized
            
            efficient_portfolios.append({
                'target_return': target * 252,  # Annualized
                'volatility': portfolio_volatility,
                'weights': weights,
                'sharpe_ratio': (target * 252 - 0.02) / portfolio_volatility
            })
        
        except:
            continue  # Skip if optimization fails
    
    return efficient_portfolios
```

**NumPy Concepts:**
- `np.linspace()` for creating target return range
- Matrix operations for portfolio calculations
- Error handling in optimization

### Step 9: Risk Attribution
```python
def calculate_risk_attribution(self, weights):
    """Calculate risk contribution of each asset"""
    
    # Portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Marginal risk contribution
    marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_volatility
    
    # Component risk contribution
    component_contrib = weights * marginal_contrib
    
    # Percentage risk contribution
    percent_contrib = component_contrib / portfolio_volatility * 100
    
    return {
        'marginal_contribution': marginal_contrib,
        'component_contribution': component_contrib,
        'percentage_contribution': percent_contrib,
        'portfolio_volatility': portfolio_volatility
    }
```

**NumPy Concepts:**
- Matrix-vector multiplication
- Element-wise operations
- Risk decomposition mathematics

### Step 10: Backtesting Framework
```python
def backtest_strategy(self, returns, rebalance_frequency=21):
    """Backtest portfolio strategy with periodic rebalancing"""
    
    n_periods = len(returns) // rebalance_frequency
    portfolio_returns = []
    portfolio_weights_history = []
    
    for period in range(n_periods):
        start_idx = period * rebalance_frequency
        end_idx = min((period + 1) * rebalance_frequency, len(returns))
        
        # Use historical data up to rebalancing date for optimization
        historical_returns = returns[:start_idx] if start_idx > 0 else returns[:rebalance_frequency]
        
        # Calculate statistics from historical data
        period_mean_returns = np.mean(historical_returns, axis=0)
        period_cov_matrix = np.cov(historical_returns.T)
        
        # Optimize portfolio (example: minimum variance)
        try:
            inv_cov = np.linalg.inv(period_cov_matrix)
            ones = np.ones((self.n_assets, 1))
            optimal_weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
            optimal_weights = optimal_weights.flatten()
        except:
            # Fallback to equal weights if optimization fails
            optimal_weights = np.ones(self.n_assets) / self.n_assets
        
        # Calculate returns for this period
        period_returns = returns[start_idx:end_idx]
        period_portfolio_returns = np.dot(period_returns, optimal_weights)
        
        portfolio_returns.extend(period_portfolio_returns)
        portfolio_weights_history.append(optimal_weights)
    
    portfolio_returns = np.array(portfolio_returns)
    
    # Calculate backtest metrics
    backtest_metrics = self.calculate_performance_metrics(portfolio_returns)
    
    return {
        'portfolio_returns': portfolio_returns,
        'weights_history': portfolio_weights_history,
        'metrics': backtest_metrics
    }
```

**NumPy Concepts:**
- Array slicing for time periods
- Dynamic portfolio rebalancing
- Historical analysis implementation

## 🎨 Sample Usage

### Basic Portfolio Analysis
```python
# Initialize analyzer
analyzer = PortfolioAnalyzer(n_assets=5, n_days=252)

# Generate market data
returns, expected_returns, cov_matrix, corr_matrix = analyzer.generate_market_data()

# Create portfolio
portfolio = analyzer.create_portfolio()
print(f"Portfolio Return: {portfolio['expected_return']*252:.2%}")
print(f"Portfolio Volatility: {portfolio['volatility']*np.sqrt(252):.2%}")

# Calculate performance metrics
portfolio_returns = np.dot(returns, portfolio['weights'])
metrics = analyzer.calculate_performance_metrics(portfolio_returns)

for metric, value in metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
```

### Risk Analysis
```python
# Value at Risk analysis
var_results = analyzer.calculate_var(portfolio_returns)
print("Value at Risk (95% confidence):")
for method, var_value in var_results.items():
    print(f"  {method}: {var_value:.4f}")

# Monte Carlo simulation
sim_stats, final_values, cum_returns = analyzer.monte_carlo_simulation()
print(f"Probability of Loss: {sim_stats['probability_loss']:.2%}")
print(f"Expected Shortfall: {sim_stats['expected_shortfall']:.4f}")
```

### Portfolio Optimization
```python
# Calculate efficient frontier
efficient_portfolios = analyzer.calculate_efficient_frontier()

# Find maximum Sharpe ratio portfolio
max_sharpe_weights = analyzer.optimize_portfolio(method='max_sharpe')
print("Maximum Sharpe Ratio Portfolio:")
for i, weight in enumerate(max_sharpe_weights):
    print(f"  {analyzer.asset_names[i]}: {weight:.2%}")
```

## 📈 Performance Tips

### 1. Matrix Operations Optimization
```python
# Use NumPy's optimized linear algebra
# Avoid loops - use vectorized operations
portfolio_returns = np.dot(returns, weights)  # Fast
# Instead of: sum(returns[i] * weights[i] for i in range(len(weights)))  # Slow
```

### 2. Memory Management
```python
# Pre-allocate arrays for large simulations
results = np.zeros((n_simulations, n_assets))
# Instead of: results = []  # Slow for large arrays
```

### 3. Numerical Stability
```python
# Check matrix condition before inversion
cond_num = np.linalg.cond(cov_matrix)
if cond_num > 1e12:
    # Use pseudo-inverse for ill-conditioned matrices
    inv_cov = np.linalg.pinv(cov_matrix)
else:
    inv_cov = np.linalg.inv(cov_matrix)
```

## 🔍 Common Issues and Solutions

### Issue 1: Singular Covariance Matrix
```python
# Problem: Matrix is not invertible
# Solution: Add regularization or use pseudo-inverse
regularized_cov = cov_matrix + np.eye(n_assets) * 1e-8
inv_cov = np.linalg.inv(regularized_cov)
```

### Issue 2: Negative Weights in Optimization
```python
# Problem: Optimization produces negative weights
# Solution: Apply constraints
weights = np.maximum(weights, 0)  # Long-only constraint
weights = weights / np.sum(weights)  # Renormalize
```

### Issue 3: Numerical Precision
```python
# Problem: Floating point precision errors
# Solution: Use appropriate tolerances
if np.abs(np.sum(weights) - 1.0) > 1e-10:
    weights = weights / np.sum(weights)
```

## 🎯 Practice Exercises

1. **Implement Black-Litterman model** for expected return estimation
2. **Create risk parity portfolio** where each asset contributes equally to risk
3. **Build factor model** using principal component analysis
4. **Implement dynamic hedging** strategy using options
5. **Create regime-switching model** for different market conditions

## 📚 Next Steps

After mastering this project:
1. Move to Project 4: Physics Simulation
2. Explore advanced optimization techniques (genetic algorithms, particle swarm)
3. Learn about alternative risk measures (CVaR, Expected Shortfall)
4. Study high-frequency trading and market microstructure
5. Implement machine learning for return prediction

This comprehensive guide provides everything needed to master financial portfolio analysis with NumPy through practical quantitative finance applications!