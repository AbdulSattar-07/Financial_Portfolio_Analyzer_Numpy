"""
Project 3: Financial Portfolio Analyzer with NumPy
Advanced quantitative finance and portfolio optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import warnings
import os
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """Advanced Financial Portfolio Analysis using NumPy"""
    
    def __init__(self, n_assets=5, n_days=252):
        self.n_assets = n_assets
        self.n_days = n_days
        self.asset_names = [f'Stock_{chr(65+i)}' for i in range(n_assets)]  # Stock_A, Stock_B, etc.
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Initialize data containers
        self.returns = None
        self.expected_returns = None
        self.cov_matrix = None
        self.correlation_matrix = None
        
        # Create directories
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs('sample_data', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        print("📁 Created directories: sample_data/, results/")
    
    # ==================== DATA GENERATION ====================
    
    def create_correlation_matrix(self, correlation_strength=0.3):
        """Create realistic correlation matrix"""
        # Start with identity matrix
        corr_matrix = np.eye(self.n_assets)
        
        # Add random correlations
        np.random.seed(42)
        for i in range(self.n_assets):
            for j in range(i+1, self.n_assets):
                # Random correlation between -0.5 and 0.7
                correlation = np.random.uniform(-0.5, 0.7) * correlation_strength
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eig(corr_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
        corr_matrix = np.dot(eigenvecs, np.dot(np.diag(eigenvals), eigenvecs.T))
        
        # Normalize diagonal to 1
        diag_sqrt = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        return corr_matrix
    
    def generate_market_data(self):
        """Generate realistic market data with correlations"""
        print("📊 Generating market data...")
        np.random.seed(42)
        
        # Define asset characteristics
        asset_data = {
            'Stock_A': {'expected_return': 0.08, 'volatility': 0.15},  # Large cap
            'Stock_B': {'expected_return': 0.12, 'volatility': 0.25},  # Growth stock
            'Stock_C': {'expected_return': 0.10, 'volatility': 0.20},  # Mid cap
            'Stock_D': {'expected_return': 0.15, 'volatility': 0.30},  # Small cap
            'Stock_E': {'expected_return': 0.06, 'volatility': 0.10}   # Defensive stock
        }
        
        # Extract expected returns and volatilities
        expected_returns = np.array([asset_data[name]['expected_return'] for name in self.asset_names])
        volatilities = np.array([asset_data[name]['volatility'] for name in self.asset_names])
        
        # Convert to daily
        daily_returns = expected_returns / 252
        daily_volatilities = volatilities / np.sqrt(252)
        
        # Create correlation matrix
        correlation_matrix = self.create_correlation_matrix()
        
        # Create covariance matrix
        cov_matrix = np.outer(daily_volatilities, daily_volatilities) * correlation_matrix
        
        # Generate correlated returns using Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix)
        independent_returns = np.random.normal(0, 1, (self.n_days, self.n_assets))
        correlated_returns = daily_returns + np.dot(independent_returns, L.T)
        
        # Store data
        self.returns = correlated_returns
        self.expected_returns = daily_returns
        self.cov_matrix = cov_matrix
        self.correlation_matrix = correlation_matrix
        
        print(f"✅ Generated {self.n_days} days of data for {self.n_assets} assets")
        return correlated_returns, daily_returns, cov_matrix, correlation_matrix
    
    def generate_market_index(self):
        """Generate market index for beta calculations"""
        np.random.seed(123)  # Different seed for market
        market_return = 0.10 / 252  # 10% annual return
        market_volatility = 0.16 / np.sqrt(252)  # 16% annual volatility
        
        market_returns = np.random.normal(market_return, market_volatility, self.n_days)
        return market_returns
    
    # ==================== PORTFOLIO CONSTRUCTION ====================
    
    def create_portfolio(self, weights=None, portfolio_name="Equal Weight"):
        """Create portfolio with given weights"""
        if weights is None:
            weights = np.ones(self.n_assets) / self.n_assets
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Annualized metrics
        annual_return = portfolio_return * 252
        annual_volatility = portfolio_volatility * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        portfolio_info = {
            'name': portfolio_name,
            'weights': weights,
            'expected_return_daily': portfolio_return,
            'expected_return_annual': annual_return,
            'volatility_daily': portfolio_volatility,
            'volatility_annual': annual_volatility,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio
        }
        
        return portfolio_info
    
    def calculate_portfolio_returns(self, weights):
        """Calculate historical portfolio returns"""
        return np.dot(self.returns, weights)
    
    # ==================== RISK ANALYSIS ====================
    
    def calculate_var(self, returns, confidence_levels=[0.95, 0.99]):
        """Calculate Value at Risk using multiple methods"""
        var_results = {}
        
        for confidence_level in confidence_levels:
            # Historical VaR
            var_historical = np.percentile(returns, (1-confidence_level)*100)
            
            # Parametric VaR (normal distribution)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(1-confidence_level)
            var_parametric = mean_return + z_score * std_return
            
            # Monte Carlo VaR
            n_simulations = 10000
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            var_monte_carlo = np.percentile(simulated_returns, (1-confidence_level)*100)
            
            var_results[f'{confidence_level:.0%}'] = {
                'historical': var_historical,
                'parametric': var_parametric,
                'monte_carlo': var_monte_carlo
            }
        
        return var_results
    
    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        """Calculate Expected Shortfall (Conditional VaR)"""
        var_threshold = np.percentile(returns, (1-confidence_level)*100)
        tail_losses = returns[returns <= var_threshold]
        expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
        
        return expected_shortfall
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown and drawdown series"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Find drawdown periods
        drawdown_start = np.where(drawdown == 0)[0]
        drawdown_end = np.where(np.diff(np.concatenate(([False], drawdown == 0, [True]))))[0]
        
        return {
            'max_drawdown': max_drawdown,
            'drawdown_series': drawdown,
            'cumulative_returns': cumulative_returns,
            'running_max': running_max
        }
    
    # ==================== PERFORMANCE METRICS ====================
    
    def calculate_performance_metrics(self, returns, benchmark_returns=None):
        """Calculate comprehensive performance metrics"""
        
        # Basic statistics
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Annualized metrics
        annual_return = mean_return * 252
        annual_volatility = volatility * np.sqrt(252)
        
        # Risk-adjusted metrics
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Sortino ratio (downside risk only)
        negative_returns = returns[returns < daily_rf]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns) * np.sqrt(252)
            sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation
        else:
            sortino_ratio = np.inf
        
        # Maximum drawdown
        drawdown_info = self.calculate_max_drawdown(returns)
        max_drawdown = drawdown_info['max_drawdown']
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # VaR and Expected Shortfall
        var_95 = np.percentile(returns, 5)
        expected_shortfall = self.calculate_expected_shortfall(returns)
        
        # Higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'expected_shortfall': expected_shortfall,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns) * np.sqrt(252)
            information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252) if np.std(active_returns) > 0 else 0
            
            # Beta calculation
            beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            
            # Alpha (Jensen's alpha)
            benchmark_annual = np.mean(benchmark_returns) * 252
            alpha = annual_return - (self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate))
            
            metrics.update({
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha
            })
        
        return metrics
    
    # ==================== CORRELATION ANALYSIS ====================
    
    def analyze_correlations(self):
        """Comprehensive correlation analysis"""
        
        # Calculate correlation matrix from returns
        corr_matrix = np.corrcoef(self.returns.T)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(self.n_assets):
            for j in range(i+1, self.n_assets):
                corr_value = corr_matrix[i, j]
                if abs(corr_value) > 0.7:
                    high_corr_pairs.append({
                        'asset1': self.asset_names[i],
                        'asset2': self.asset_names[j],
                        'correlation': corr_value
                    })
        
        # Calculate average correlations
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        avg_correlation = np.mean(upper_triangle)
        max_correlation = np.max(upper_triangle)
        min_correlation = np.min(upper_triangle)
        
        # Diversification ratio
        equal_weights = np.ones(self.n_assets) / self.n_assets
        individual_volatilities = np.sqrt(np.diag(self.cov_matrix)) * np.sqrt(252)
        weighted_avg_vol = np.dot(equal_weights, individual_volatilities)
        portfolio_vol = np.sqrt(np.dot(equal_weights.T, np.dot(self.cov_matrix, equal_weights))) * np.sqrt(252)
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs,
            'average_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'min_correlation': min_correlation,
            'diversification_ratio': diversification_ratio
        }
    
    # ==================== MONTE CARLO SIMULATION ====================
    
    def monte_carlo_simulation(self, weights=None, n_simulations=10000, time_horizon=252):
        """Run Monte Carlo simulation for portfolio scenarios"""
        
        if weights is None:
            weights = np.ones(self.n_assets) / self.n_assets
        
        print(f"🎲 Running Monte Carlo simulation ({n_simulations:,} scenarios)...")
        
        # Generate random scenarios using multivariate normal distribution
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
        
        # Calculate path-dependent metrics
        max_drawdowns = []
        for i in range(n_simulations):
            path_returns = cumulative_returns[i, :]
            running_max = np.maximum.accumulate(path_returns)
            drawdown = (path_returns - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))
        
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate comprehensive statistics
        simulation_stats = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'percentile_1': np.percentile(final_values, 1),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_95': np.percentile(final_values, 95),
            'percentile_99': np.percentile(final_values, 99),
            'probability_loss': np.mean(final_values < 1.0),
            'probability_loss_10': np.mean(final_values < 0.9),
            'probability_loss_20': np.mean(final_values < 0.8),
            'expected_shortfall_5': np.mean(final_values[final_values <= np.percentile(final_values, 5)]),
            'max_drawdown_mean': np.mean(max_drawdowns),
            'max_drawdown_worst': np.min(max_drawdowns)
        }
        
        print("✅ Monte Carlo simulation completed")
        
        return simulation_stats, final_values, cumulative_returns
    
    # ==================== PORTFOLIO OPTIMIZATION ====================
    
    def optimize_portfolio(self, method='min_variance', target_return=None, constraints=None):
        """Portfolio optimization using various methods"""
        
        if constraints is None:
            constraints = {'long_only': True, 'max_weight': 1.0}
        
        if method == 'min_variance':
            return self._optimize_min_variance(constraints)
        elif method == 'max_sharpe':
            return self._optimize_max_sharpe(constraints)
        elif method == 'target_return':
            if target_return is None:
                raise ValueError("Target return must be specified for target_return method")
            return self._optimize_target_return(target_return, constraints)
        elif method == 'risk_parity':
            return self._optimize_risk_parity(constraints)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_min_variance(self, constraints):
        """Minimum variance portfolio optimization"""
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
            ones = np.ones((self.n_assets, 1))
            
            # Analytical solution: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
            numerator = np.dot(inv_cov, ones)
            denominator = np.dot(ones.T, numerator)
            optimal_weights = (numerator / denominator).flatten()
            
            # Apply constraints
            if constraints.get('long_only', True):
                optimal_weights = np.maximum(optimal_weights, 0)
                optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            return optimal_weights
            
        except np.linalg.LinAlgError:
            print("⚠️ Covariance matrix is singular, using regularization")
            regularized_cov = self.cov_matrix + np.eye(self.n_assets) * 1e-8
            inv_cov = np.linalg.inv(regularized_cov)
            ones = np.ones((self.n_assets, 1))
            numerator = np.dot(inv_cov, ones)
            denominator = np.dot(ones.T, numerator)
            optimal_weights = (numerator / denominator).flatten()
            
            if constraints.get('long_only', True):
                optimal_weights = np.maximum(optimal_weights, 0)
                optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            return optimal_weights
    
    def _optimize_max_sharpe(self, constraints):
        """Maximum Sharpe ratio portfolio optimization"""
        try:
            daily_rf = self.risk_free_rate / 252
            excess_returns = self.expected_returns - daily_rf
            
            inv_cov = np.linalg.inv(self.cov_matrix)
            ones = np.ones(self.n_assets)
            
            # Analytical solution for tangency portfolio
            numerator = np.dot(inv_cov, excess_returns)
            denominator = np.dot(ones, numerator)
            optimal_weights = numerator / denominator
            
            # Apply constraints
            if constraints.get('long_only', True):
                optimal_weights = np.maximum(optimal_weights, 0)
                optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            return optimal_weights
            
        except np.linalg.LinAlgError:
            print("⚠️ Using regularized covariance matrix")
            regularized_cov = self.cov_matrix + np.eye(self.n_assets) * 1e-8
            inv_cov = np.linalg.inv(regularized_cov)
            daily_rf = self.risk_free_rate / 252
            excess_returns = self.expected_returns - daily_rf
            ones = np.ones(self.n_assets)
            numerator = np.dot(inv_cov, excess_returns)
            denominator = np.dot(ones, numerator)
            optimal_weights = numerator / denominator
            
            if constraints.get('long_only', True):
                optimal_weights = np.maximum(optimal_weights, 0)
                optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            return optimal_weights
    
    def _optimize_target_return(self, target_return, constraints):
        """Target return portfolio optimization"""
        try:
            # Convert annual target return to daily
            daily_target = target_return / 252
            
            inv_cov = np.linalg.inv(self.cov_matrix)
            ones = np.ones((self.n_assets, 1))
            mu = self.expected_returns.reshape(-1, 1)
            
            # Lagrange multiplier method
            A = np.dot(mu.T, np.dot(inv_cov, mu))[0, 0]
            B = np.dot(mu.T, np.dot(inv_cov, ones))[0, 0]
            C = np.dot(ones.T, np.dot(inv_cov, ones))[0, 0]
            
            denominator = A * C - B**2
            if abs(denominator) < 1e-10:
                print("⚠️ Optimization problem is ill-conditioned")
                return np.ones(self.n_assets) / self.n_assets
            
            lambda_1 = (C * daily_target - B) / denominator
            lambda_2 = (B * daily_target - A) / denominator
            
            optimal_weights = (lambda_1 * np.dot(inv_cov, mu) + lambda_2 * np.dot(inv_cov, ones)).flatten()
            
            # Apply constraints
            if constraints.get('long_only', True):
                optimal_weights = np.maximum(optimal_weights, 0)
                optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            return optimal_weights
            
        except (np.linalg.LinAlgError, ValueError):
            print("⚠️ Optimization failed, returning equal weights")
            return np.ones(self.n_assets) / self.n_assets
    
    def _optimize_risk_parity(self, constraints):
        """Risk parity portfolio optimization"""
        def risk_budget_objective(weights):
            """Objective function for risk parity"""
            weights = weights / np.sum(weights)  # Normalize
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / self.n_assets  # Equal risk contribution
            return np.sum((contrib - target_contrib)**2)
        
        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if constraints.get('long_only', True):
            bounds = [(0, constraints.get('max_weight', 1.0)) for _ in range(self.n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(self.n_assets)]
        
        # Optimize
        result = optimize.minimize(risk_budget_objective, x0, method='SLSQP', 
                                 bounds=bounds, constraints=cons)
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # Ensure normalization
            return optimal_weights
        else:
            print("⚠️ Risk parity optimization failed, returning equal weights")
            return np.ones(self.n_assets) / self.n_assets
    
    # ==================== EFFICIENT FRONTIER ====================
    
    def calculate_efficient_frontier(self, n_points=50):
        """Calculate the efficient frontier"""
        
        print("📈 Calculating efficient frontier...")
        
        # Define range of target returns
        min_return = np.min(self.expected_returns) * 252  # Annualized
        max_return = np.max(self.expected_returns) * 252  # Annualized
        
        # Extend range slightly
        return_range = max_return - min_return
        min_return = min_return - 0.1 * return_range
        max_return = max_return + 0.1 * return_range
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_portfolios = []
        
        for target in target_returns:
            try:
                weights = self.optimize_portfolio(method='target_return', target_return=target)
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(weights, self.expected_returns) * 252  # Annualized
                portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualized
                
                # Sharpe ratio
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                efficient_portfolios.append({
                    'target_return': target,
                    'actual_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'weights': weights,
                    'sharpe_ratio': sharpe_ratio
                })
            
            except:
                continue  # Skip if optimization fails
        
        print(f"✅ Calculated {len(efficient_portfolios)} efficient portfolios")
        return efficient_portfolios
    
    # ==================== RISK ATTRIBUTION ====================
    
    def calculate_risk_attribution(self, weights):
        """Calculate risk contribution of each asset"""
        
        # Portfolio variance and volatility
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Marginal risk contribution (partial derivative of portfolio volatility w.r.t. weights)
        marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_volatility
        
        # Component risk contribution (weight × marginal contribution)
        component_contrib = weights * marginal_contrib
        
        # Percentage risk contribution
        percent_contrib = component_contrib / portfolio_volatility * 100
        
        return {
            'portfolio_volatility': portfolio_volatility * np.sqrt(252),  # Annualized
            'marginal_contribution': marginal_contrib * np.sqrt(252),  # Annualized
            'component_contribution': component_contrib * np.sqrt(252),  # Annualized
            'percentage_contribution': percent_contrib,
            'weights': weights * 100  # As percentages
        }
    
    # ==================== BACKTESTING ====================
    
    def backtest_strategy(self, strategy='min_variance', rebalance_frequency=21, lookback_window=63):
        """Backtest portfolio strategy with periodic rebalancing"""
        
        print(f"🔄 Backtesting {strategy} strategy...")
        print(f"   Rebalancing every {rebalance_frequency} days")
        print(f"   Using {lookback_window} days lookback window")
        
        n_periods = (len(self.returns) - lookback_window) // rebalance_frequency
        portfolio_returns = []
        portfolio_weights_history = []
        rebalance_dates = []
        
        for period in range(n_periods):
            # Calculate start and end indices for this period
            lookback_start = period * rebalance_frequency
            lookback_end = lookback_start + lookback_window
            period_start = lookback_end
            period_end = min(period_start + rebalance_frequency, len(self.returns))
            
            if period_end <= period_start:
                break
            
            # Use historical data for optimization
            historical_returns = self.returns[lookback_start:lookback_end]
            
            # Calculate statistics from historical data
            period_expected_returns = np.mean(historical_returns, axis=0)
            period_cov_matrix = np.cov(historical_returns.T)
            
            # Temporarily update analyzer with period data
            original_expected_returns = self.expected_returns.copy()
            original_cov_matrix = self.cov_matrix.copy()
            
            self.expected_returns = period_expected_returns
            self.cov_matrix = period_cov_matrix
            
            # Optimize portfolio
            try:
                optimal_weights = self.optimize_portfolio(method=strategy)
            except:
                # Fallback to equal weights if optimization fails
                optimal_weights = np.ones(self.n_assets) / self.n_assets
            
            # Restore original data
            self.expected_returns = original_expected_returns
            self.cov_matrix = original_cov_matrix
            
            # Calculate returns for this period using optimized weights
            period_returns = self.returns[period_start:period_end]
            period_portfolio_returns = np.dot(period_returns, optimal_weights)
            
            portfolio_returns.extend(period_portfolio_returns)
            portfolio_weights_history.append(optimal_weights)
            rebalance_dates.append(period_start)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate backtest metrics
        backtest_metrics = self.calculate_performance_metrics(portfolio_returns)
        
        # Calculate benchmark (buy and hold equal weight)
        equal_weights = np.ones(self.n_assets) / self.n_assets
        benchmark_returns = np.dot(self.returns[:len(portfolio_returns)], equal_weights)
        benchmark_metrics = self.calculate_performance_metrics(benchmark_returns)
        
        print("✅ Backtesting completed")
        
        return {
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'weights_history': portfolio_weights_history,
            'rebalance_dates': rebalance_dates,
            'strategy_metrics': backtest_metrics,
            'benchmark_metrics': benchmark_metrics
        }
    
    # ==================== ANALYSIS AND REPORTING ====================
    
    def generate_comprehensive_report(self):
        """Generate comprehensive portfolio analysis report"""
        
        print("📋 Generating comprehensive analysis report...")
        
        # Ensure data is generated
        if self.returns is None:
            self.generate_market_data()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period': f"{self.n_days} days",
            'assets': self.asset_names,
            'risk_free_rate': self.risk_free_rate
        }
        
        # 1. Market Data Analysis
        market_stats = {
            'individual_returns': (self.expected_returns * 252).tolist(),
            'individual_volatilities': (np.sqrt(np.diag(self.cov_matrix)) * np.sqrt(252)).tolist(),
            'correlation_analysis': self.analyze_correlations()
        }
        report['market_analysis'] = market_stats
        
        # 2. Portfolio Strategies Comparison
        strategies = ['min_variance', 'max_sharpe', 'risk_parity']
        strategy_results = {}
        
        for strategy in strategies:
            try:
                weights = self.optimize_portfolio(method=strategy)
                portfolio_info = self.create_portfolio(weights, strategy.replace('_', ' ').title())
                portfolio_returns = self.calculate_portfolio_returns(weights)
                performance_metrics = self.calculate_performance_metrics(portfolio_returns)
                risk_attribution = self.calculate_risk_attribution(weights)
                
                strategy_results[strategy] = {
                    'portfolio_info': portfolio_info,
                    'performance_metrics': performance_metrics,
                    'risk_attribution': risk_attribution
                }
            except Exception as e:
                print(f"⚠️ Error analyzing {strategy}: {e}")
                continue
        
        report['strategy_comparison'] = strategy_results
        
        # 3. Risk Analysis
        equal_weights = np.ones(self.n_assets) / self.n_assets
        equal_weight_returns = self.calculate_portfolio_returns(equal_weights)
        
        risk_analysis = {
            'var_analysis': self.calculate_var(equal_weight_returns),
            'expected_shortfall': self.calculate_expected_shortfall(equal_weight_returns),
            'max_drawdown_analysis': self.calculate_max_drawdown(equal_weight_returns)
        }
        report['risk_analysis'] = risk_analysis
        
        # 4. Monte Carlo Simulation
        mc_stats, _, _ = self.monte_carlo_simulation(equal_weights, n_simulations=5000)
        report['monte_carlo_simulation'] = mc_stats
        
        # 5. Efficient Frontier
        efficient_portfolios = self.calculate_efficient_frontier(n_points=20)
        report['efficient_frontier'] = efficient_portfolios
        
        # 6. Backtesting Results
        backtest_results = self.backtest_strategy('min_variance', rebalance_frequency=21)
        report['backtesting'] = {
            'strategy_metrics': backtest_results['strategy_metrics'],
            'benchmark_metrics': backtest_results['benchmark_metrics']
        }
        
        # Save report
        report_filename = f"results/portfolio_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(report), f, indent=2)
        
        print(f"✅ Report saved to {report_filename}")
        return report
    
    def create_visualization_summary(self):
        """Create visualization summary of key results"""
        
        print("📊 Creating visualization summary...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Portfolio Analysis Summary', fontsize=16, fontweight='bold')
        
        # 1. Asset Returns and Volatilities
        annual_returns = self.expected_returns * 252
        annual_volatilities = np.sqrt(np.diag(self.cov_matrix)) * np.sqrt(252)
        
        axes[0, 0].scatter(annual_volatilities, annual_returns, s=100, alpha=0.7)
        for i, name in enumerate(self.asset_names):
            axes[0, 0].annotate(name, (annual_volatilities[i], annual_returns[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[0, 0].set_xlabel('Annual Volatility')
        axes[0, 0].set_ylabel('Annual Return')
        axes[0, 0].set_title('Risk-Return Profile of Assets')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Correlation Heatmap
        corr_analysis = self.analyze_correlations()
        im = axes[0, 1].imshow(corr_analysis['correlation_matrix'], cmap='RdBu', vmin=-1, vmax=1)
        axes[0, 1].set_xticks(range(self.n_assets))
        axes[0, 1].set_yticks(range(self.n_assets))
        axes[0, 1].set_xticklabels(self.asset_names, rotation=45)
        axes[0, 1].set_yticklabels(self.asset_names)
        axes[0, 1].set_title('Asset Correlation Matrix')
        
        # Add correlation values to heatmap
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                text = axes[0, 1].text(j, i, f'{corr_analysis["correlation_matrix"][i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        # 3. Efficient Frontier
        efficient_portfolios = self.calculate_efficient_frontier(n_points=30)
        if efficient_portfolios:
            returns = [p['actual_return'] for p in efficient_portfolios]
            volatilities = [p['volatility'] for p in efficient_portfolios]
            
            axes[0, 2].plot(volatilities, returns, 'b-', linewidth=2, label='Efficient Frontier')
            
            # Mark special portfolios
            min_var_weights = self.optimize_portfolio('min_variance')
            min_var_return = np.dot(min_var_weights, self.expected_returns) * 252
            min_var_vol = np.sqrt(np.dot(min_var_weights.T, np.dot(self.cov_matrix, min_var_weights))) * np.sqrt(252)
            axes[0, 2].scatter(min_var_vol, min_var_return, color='red', s=100, label='Min Variance', zorder=5)
            
            max_sharpe_weights = self.optimize_portfolio('max_sharpe')
            max_sharpe_return = np.dot(max_sharpe_weights, self.expected_returns) * 252
            max_sharpe_vol = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(self.cov_matrix, max_sharpe_weights))) * np.sqrt(252)
            axes[0, 2].scatter(max_sharpe_vol, max_sharpe_return, color='green', s=100, label='Max Sharpe', zorder=5)
            
            axes[0, 2].set_xlabel('Annual Volatility')
            axes[0, 2].set_ylabel('Annual Return')
            axes[0, 2].set_title('Efficient Frontier')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Portfolio Weights Comparison
        strategies = ['min_variance', 'max_sharpe', 'risk_parity']
        strategy_weights = []
        strategy_labels = []
        
        for strategy in strategies:
            try:
                weights = self.optimize_portfolio(method=strategy)
                strategy_weights.append(weights)
                strategy_labels.append(strategy.replace('_', ' ').title())
            except:
                continue
        
        if strategy_weights:
            x = np.arange(self.n_assets)
            width = 0.25
            
            for i, (weights, label) in enumerate(zip(strategy_weights, strategy_labels)):
                axes[1, 0].bar(x + i*width, weights, width, label=label, alpha=0.8)
            
            axes[1, 0].set_xlabel('Assets')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_title('Portfolio Weights Comparison')
            axes[1, 0].set_xticks(x + width)
            axes[1, 0].set_xticklabels(self.asset_names)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Monte Carlo Simulation Results
        equal_weights = np.ones(self.n_assets) / self.n_assets
        mc_stats, final_values, _ = self.monte_carlo_simulation(equal_weights, n_simulations=5000)
        
        axes[1, 1].hist(final_values, bins=50, alpha=0.7, density=True)
        axes[1, 1].axvline(1.0, color='red', linestyle='--', label='Break-even')
        axes[1, 1].axvline(mc_stats['percentile_5'], color='orange', linestyle='--', label='5th Percentile')
        axes[1, 1].axvline(mc_stats['percentile_95'], color='green', linestyle='--', label='95th Percentile')
        axes[1, 1].set_xlabel('Final Portfolio Value')
        axes[1, 1].set_ylabel('Probability Density')
        axes[1, 1].set_title('Monte Carlo Simulation Results')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance Metrics Comparison
        metrics_comparison = {}
        for strategy in strategies:
            try:
                weights = self.optimize_portfolio(method=strategy)
                returns = self.calculate_portfolio_returns(weights)
                metrics = self.calculate_performance_metrics(returns)
                metrics_comparison[strategy.replace('_', ' ').title()] = metrics
            except:
                continue
        
        if metrics_comparison:
            metric_names = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
            metric_labels = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown']
            
            x = np.arange(len(metric_names))
            width = 0.25
            
            for i, (strategy, metrics) in enumerate(metrics_comparison.items()):
                values = [abs(metrics[name]) if name == 'max_drawdown' else metrics[name] for name in metric_names]
                axes[1, 2].bar(x + i*width, values, width, label=strategy, alpha=0.8)
            
            axes[1, 2].set_xlabel('Metrics')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].set_title('Performance Metrics Comparison')
            axes[1, 2].set_xticks(x + width)
            axes[1, 2].set_xticklabels(metric_labels, rotation=45)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_filename = f"results/portfolio_analysis_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to {viz_filename}")

def main():
    """Main function to run the portfolio analysis demonstration"""
    print("💰 FINANCIAL PORTFOLIO ANALYZER WITH NUMPY")
    print("=" * 50)
    print("Advanced quantitative finance and portfolio optimization")
    print("Learning linear algebra and statistical analysis\n")
    
    # Create analyzer instance
    analyzer = PortfolioAnalyzer(n_assets=5, n_days=252)
    
    # Generate market data
    analyzer.generate_market_data()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Create visualizations
    analyzer.create_visualization_summary()
    
    print("\n✅ PROJECT COMPLETE!")
    print("Key NumPy concepts learned:")
    print("- Matrix operations and linear algebra")
    print("- Advanced statistical analysis")
    print("- Monte Carlo simulation techniques")
    print("- Correlation and covariance analysis")
    print("- Portfolio optimization algorithms")
    print("- Risk metrics and performance indicators")
    print("- Vectorized operations for performance")
    print("- Financial mathematics implementation")

if __name__ == "__main__":
    main()