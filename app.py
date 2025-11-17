"""
Advanced Financial Portfolio Analyzer - Streamlit UI
Interactive interface for NumPy-based portfolio analysis and optimization
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats, optimize
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="💰 Portfolio Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        animation: slideInDown 1s ease-out;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.3);
    }
    
    /* Animation keyframes */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1e3c72;
        margin-bottom: 1rem;
        animation: fadeInUp 0.8s ease-out;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Control panel styling */
    .control-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1e3c72;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(30, 60, 114, 0.4);
        animation: pulse 1s infinite;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Success/warning styling */
    .success-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            font-size: 0.9rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class StreamlitPortfolioAnalyzer:
    """Streamlit-based Portfolio Analysis Interface"""
    
    def __init__(self):
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'market_data_generated' not in st.session_state:
            st.session_state.market_data_generated = False
    
    def create_correlation_matrix(self, n_assets, correlation_strength=0.3):
        """Create realistic correlation matrix"""
        np.random.seed(42)
        corr_matrix = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                correlation = np.random.uniform(-0.5, 0.7) * correlation_strength
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eig(corr_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        corr_matrix = np.dot(eigenvecs, np.dot(np.diag(eigenvals), eigenvecs.T))
        
        # Normalize diagonal to 1
        diag_sqrt = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        return corr_matrix
    
    def generate_market_data(self, n_assets, n_days, asset_params):
        """Generate realistic market data"""
        np.random.seed(42)
        
        # Extract parameters
        expected_returns = np.array([params['expected_return'] for params in asset_params.values()])
        volatilities = np.array([params['volatility'] for params in asset_params.values()])
        
        # Convert to daily
        daily_returns = expected_returns / 252
        daily_volatilities = volatilities / np.sqrt(252)
        
        # Create correlation and covariance matrices
        correlation_matrix = self.create_correlation_matrix(n_assets)
        cov_matrix = np.outer(daily_volatilities, daily_volatilities) * correlation_matrix
        
        # Generate correlated returns
        L = np.linalg.cholesky(cov_matrix)
        independent_returns = np.random.normal(0, 1, (n_days, n_assets))
        correlated_returns = daily_returns + np.dot(independent_returns, L.T)
        
        return {
            'returns': correlated_returns,
            'expected_returns': daily_returns,
            'cov_matrix': cov_matrix,
            'correlation_matrix': correlation_matrix,
            'asset_names': list(asset_params.keys()),
            'annual_returns': expected_returns,
            'annual_volatilities': volatilities
        }
    
    def optimize_portfolio(self, data, method='min_variance', target_return=None):
        """Portfolio optimization"""
        expected_returns = data['expected_returns']
        cov_matrix = data['cov_matrix']
        n_assets = len(expected_returns)
        
        try:
            if method == 'min_variance':
                inv_cov = np.linalg.inv(cov_matrix)
                ones = np.ones((n_assets, 1))
                numerator = np.dot(inv_cov, ones)
                denominator = np.dot(ones.T, numerator)
                weights = (numerator / denominator).flatten()
            
            elif method == 'max_sharpe':
                risk_free_rate = 0.02 / 252
                excess_returns = expected_returns - risk_free_rate
                inv_cov = np.linalg.inv(cov_matrix)
                ones = np.ones(n_assets)
                numerator = np.dot(inv_cov, excess_returns)
                denominator = np.dot(ones, numerator)
                weights = numerator / denominator
            
            elif method == 'target_return' and target_return is not None:
                daily_target = target_return / 252
                inv_cov = np.linalg.inv(cov_matrix)
                ones = np.ones((n_assets, 1))
                mu = expected_returns.reshape(-1, 1)
                
                A = np.dot(mu.T, np.dot(inv_cov, mu))[0, 0]
                B = np.dot(mu.T, np.dot(inv_cov, ones))[0, 0]
                C = np.dot(ones.T, np.dot(inv_cov, ones))[0, 0]
                
                denominator = A * C - B**2
                if abs(denominator) < 1e-10:
                    return np.ones(n_assets) / n_assets
                
                lambda_1 = (C * daily_target - B) / denominator
                lambda_2 = (B * daily_target - A) / denominator
                
                weights = (lambda_1 * np.dot(inv_cov, mu) + lambda_2 * np.dot(inv_cov, ones)).flatten()
            
            else:
                return np.ones(n_assets) / n_assets
            
            # Apply constraints
            weights = np.maximum(weights, 0)  # Long-only
            weights = weights / np.sum(weights)  # Normalize
            
            return weights
            
        except np.linalg.LinAlgError:
            return np.ones(n_assets) / n_assets
    
    def calculate_portfolio_metrics(self, weights, data, risk_free_rate=0.02):
        """Calculate portfolio metrics"""
        expected_returns = data['expected_returns']
        cov_matrix = data['cov_matrix']
        
        # Portfolio return and risk
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Annualized metrics
        annual_return = portfolio_return * 252
        annual_volatility = portfolio_volatility * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        return {
            'weights': weights,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'daily_return': portfolio_return,
            'daily_volatility': portfolio_volatility
        }
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        return np.percentile(returns, (1-confidence_level)*100)
    
    def monte_carlo_simulation(self, weights, data, n_simulations=10000, time_horizon=252):
        """Monte Carlo simulation"""
        expected_returns = data['expected_returns']
        cov_matrix = data['cov_matrix']
        
        # Generate scenarios
        scenarios = np.random.multivariate_normal(
            expected_returns, cov_matrix, (n_simulations, time_horizon)
        )
        
        # Calculate portfolio returns
        portfolio_scenarios = np.dot(scenarios, weights)
        
        # Calculate final values
        cumulative_returns = np.cumprod(1 + portfolio_scenarios, axis=1)
        final_values = cumulative_returns[:, -1]
        
        return {
            'final_values': final_values,
            'mean_final_value': np.mean(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_95': np.percentile(final_values, 95),
            'probability_loss': np.mean(final_values < 1.0)
        }
    
    def calculate_efficient_frontier(self, data, n_points=50):
        """Calculate efficient frontier"""
        expected_returns = data['expected_returns']
        
        min_return = np.min(expected_returns) * 252
        max_return = np.max(expected_returns) * 252
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_portfolios = []
        for target in target_returns:
            try:
                weights = self.optimize_portfolio(data, 'target_return', target)
                metrics = self.calculate_portfolio_metrics(weights, data)
                efficient_portfolios.append({
                    'return': metrics['annual_return'],
                    'volatility': metrics['annual_volatility'],
                    'weights': weights
                })
            except:
                continue
        
        return efficient_portfolios
    
    def create_correlation_heatmap(self, correlation_matrix, asset_names):
        """Create correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=asset_names,
            y=asset_names,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            xaxis_title='Assets',
            yaxis_title='Assets',
            height=500
        )
        
        return fig
    
    def create_efficient_frontier_plot(self, efficient_portfolios, special_portfolios=None):
        """Create efficient frontier plot"""
        if not efficient_portfolios:
            return None
        
        returns = [p['return'] for p in efficient_portfolios]
        volatilities = [p['volatility'] for p in efficient_portfolios]
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=3)
        ))
        
        # Special portfolios
        if special_portfolios:
            for name, portfolio in special_portfolios.items():
                fig.add_trace(go.Scatter(
                    x=[portfolio['annual_volatility']],
                    y=[portfolio['annual_return']],
                    mode='markers',
                    name=name,
                    marker=dict(size=12)
                ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Annual Volatility',
            yaxis_title='Annual Return',
            height=500,
            hovermode='closest'
        )
        
        return fig
    
    def create_portfolio_composition_chart(self, weights, asset_names, title="Portfolio Composition"):
        """Create portfolio composition pie chart"""
        fig = go.Figure(data=[go.Pie(
            labels=asset_names,
            values=weights,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_monte_carlo_histogram(self, final_values):
        """Create Monte Carlo results histogram"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=final_values,
            nbinsx=50,
            name='Final Portfolio Values',
            opacity=0.7
        ))
        
        # Add reference lines
        fig.add_vline(x=1.0, line_dash="dash", line_color="red", 
                     annotation_text="Break-even")
        fig.add_vline(x=np.percentile(final_values, 5), line_dash="dash", 
                     line_color="orange", annotation_text="5th Percentile")
        fig.add_vline(x=np.percentile(final_values, 95), line_dash="dash", 
                     line_color="green", annotation_text="95th Percentile")
        
        fig.update_layout(
            title='Monte Carlo Simulation Results',
            xaxis_title='Final Portfolio Value',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig

def main():
    # Initialize analyzer
    analyzer = StreamlitPortfolioAnalyzer()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>💰 Advanced Portfolio Analyzer</h1>
        <p>Interactive NumPy-powered quantitative finance and portfolio optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Portfolio Configuration")
    
    # Asset configuration
    st.sidebar.markdown("### 📊 Asset Parameters")
    n_assets = st.sidebar.slider("Number of Assets", 3, 8, 5)
    n_days = st.sidebar.slider("Analysis Period (days)", 100, 500, 252)
    
    # Asset parameters
    asset_params = {}
    asset_names = [f'Asset_{chr(65+i)}' for i in range(n_assets)]
    
    with st.sidebar.expander("🔧 Customize Asset Parameters"):
        for i, name in enumerate(asset_names):
            st.markdown(f"**{name}**")
            col1, col2 = st.columns(2)
            with col1:
                expected_return = st.number_input(
                    f"Expected Return (%)", 
                    min_value=0.0, max_value=30.0, value=8.0 + i*2.0, step=0.5,
                    key=f"return_{i}"
                ) / 100
            with col2:
                volatility = st.number_input(
                    f"Volatility (%)", 
                    min_value=5.0, max_value=50.0, value=15.0 + i*3.0, step=1.0,
                    key=f"vol_{i}"
                ) / 100
            
            asset_params[name] = {
                'expected_return': expected_return,
                'volatility': volatility
            }
    
    # Generate market data
    if st.sidebar.button("🚀 Generate Market Data"):
        with st.spinner("Generating market data..."):
            st.session_state.portfolio_data = analyzer.generate_market_data(
                n_assets, n_days, asset_params
            )
            st.session_state.market_data_generated = True
        st.success("✅ Market data generated successfully!")
        st.balloons()
    
    if not st.session_state.market_data_generated:
        st.info("👆 Please generate market data from the sidebar to begin analysis!")
        return
    
    data = st.session_state.portfolio_data
    
    # Analysis options
    st.sidebar.markdown("### 📈 Analysis Options")
    show_correlation = st.sidebar.checkbox("📊 Correlation Analysis", value=True)
    show_optimization = st.sidebar.checkbox("🎯 Portfolio Optimization", value=True)
    show_efficient_frontier = st.sidebar.checkbox("📈 Efficient Frontier", value=True)
    show_monte_carlo = st.sidebar.checkbox("🎲 Monte Carlo Simulation", value=True)
    show_risk_analysis = st.sidebar.checkbox("⚠️ Risk Analysis", value=True)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview", "🎯 Portfolio Optimization", "📈 Efficient Frontier", 
        "🎲 Monte Carlo", "⚠️ Risk Analysis"
    ])
    
    with tab1:
        st.markdown("## 📊 Market Data Overview")
        
        # Asset characteristics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Expected Returns & Volatilities")
            asset_df = pd.DataFrame({
                'Asset': data['asset_names'],
                'Expected Return (%)': data['annual_returns'] * 100,
                'Volatility (%)': data['annual_volatilities'] * 100,
                'Sharpe Ratio': (data['annual_returns'] - 0.02) / data['annual_volatilities']
            })
            st.dataframe(asset_df, use_container_width=True)
        
        with col2:
            # Risk-return scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['annual_volatilities'] * 100,
                y=data['annual_returns'] * 100,
                mode='markers+text',
                text=data['asset_names'],
                textposition='top center',
                marker=dict(size=12, color='blue'),
                name='Assets'
            ))
            
            fig.update_layout(
                title='Risk-Return Profile',
                xaxis_title='Annual Volatility (%)',
                yaxis_title='Annual Return (%)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if show_correlation:
            st.markdown("### 🔗 Correlation Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                corr_fig = analyzer.create_correlation_heatmap(
                    data['correlation_matrix'], data['asset_names']
                )
                st.plotly_chart(corr_fig, use_container_width=True)
            
            with col2:
                # Correlation statistics
                upper_triangle = data['correlation_matrix'][np.triu_indices_from(data['correlation_matrix'], k=1)]
                
                st.markdown("""
                <div class="metric-card">
                    <h4>Correlation Statistics</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Average Correlation", f"{np.mean(upper_triangle):.3f}")
                st.metric("Max Correlation", f"{np.max(upper_triangle):.3f}")
                st.metric("Min Correlation", f"{np.min(upper_triangle):.3f}")
                
                # Diversification ratio
                equal_weights = np.ones(n_assets) / n_assets
                individual_vols = np.sqrt(np.diag(data['cov_matrix'])) * np.sqrt(252)
                weighted_avg_vol = np.dot(equal_weights, individual_vols)
                portfolio_vol = np.sqrt(np.dot(equal_weights.T, np.dot(data['cov_matrix'], equal_weights))) * np.sqrt(252)
                div_ratio = weighted_avg_vol / portfolio_vol
                
                st.metric("Diversification Ratio", f"{div_ratio:.3f}")
    
    with tab2:
        if show_optimization:
            st.markdown("## 🎯 Portfolio Optimization")
            
            # Optimization methods
            col1, col2, col3 = st.columns(3)
            
            strategies = {
                'Minimum Variance': 'min_variance',
                'Maximum Sharpe': 'max_sharpe',
                'Equal Weight': 'equal_weight'
            }
            
            strategy_results = {}
            
            for strategy_name, strategy_code in strategies.items():
                if strategy_code == 'equal_weight':
                    weights = np.ones(n_assets) / n_assets
                else:
                    weights = analyzer.optimize_portfolio(data, strategy_code)
                
                metrics = analyzer.calculate_portfolio_metrics(weights, data)
                strategy_results[strategy_name] = metrics
            
            # Display results
            for i, (strategy_name, metrics) in enumerate(strategy_results.items()):
                col = [col1, col2, col3][i]
                
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{strategy_name} Portfolio</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Annual Return", f"{metrics['annual_return']:.2%}")
                    st.metric("Annual Volatility", f"{metrics['annual_volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                    
                    # Portfolio composition
                    pie_fig = analyzer.create_portfolio_composition_chart(
                        metrics['weights'], data['asset_names'], f"{strategy_name} Weights"
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)
            
            # Comparison table
            st.markdown("### 📊 Strategy Comparison")
            comparison_df = pd.DataFrame({
                'Strategy': list(strategy_results.keys()),
                'Annual Return (%)': [metrics['annual_return'] * 100 for metrics in strategy_results.values()],
                'Annual Volatility (%)': [metrics['annual_volatility'] * 100 for metrics in strategy_results.values()],
                'Sharpe Ratio': [metrics['sharpe_ratio'] for metrics in strategy_results.values()]
            })
            st.dataframe(comparison_df, use_container_width=True)
    
    with tab3:
        if show_efficient_frontier:
            st.markdown("## 📈 Efficient Frontier Analysis")
            
            with st.spinner("Calculating efficient frontier..."):
                efficient_portfolios = analyzer.calculate_efficient_frontier(data)
            
            if efficient_portfolios:
                # Get special portfolios for plotting
                min_var_weights = analyzer.optimize_portfolio(data, 'min_variance')
                max_sharpe_weights = analyzer.optimize_portfolio(data, 'max_sharpe')
                
                special_portfolios = {
                    'Min Variance': analyzer.calculate_portfolio_metrics(min_var_weights, data),
                    'Max Sharpe': analyzer.calculate_portfolio_metrics(max_sharpe_weights, data)
                }
                
                # Plot efficient frontier
                ef_fig = analyzer.create_efficient_frontier_plot(efficient_portfolios, special_portfolios)
                st.plotly_chart(ef_fig, use_container_width=True)
                
                # Interactive portfolio selection
                st.markdown("### 🎯 Custom Target Return Portfolio")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    min_ret = min([p['return'] for p in efficient_portfolios])
                    max_ret = max([p['return'] for p in efficient_portfolios])
                    
                    target_return = st.slider(
                        "Target Annual Return (%)",
                        min_value=min_ret * 100,
                        max_value=max_ret * 100,
                        value=(min_ret + max_ret) * 50,
                        step=0.5
                    ) / 100
                    
                    # Calculate portfolio for target return
                    target_weights = analyzer.optimize_portfolio(data, 'target_return', target_return)
                    target_metrics = analyzer.calculate_portfolio_metrics(target_weights, data)
                    
                    st.metric("Achieved Return", f"{target_metrics['annual_return']:.2%}")
                    st.metric("Portfolio Volatility", f"{target_metrics['annual_volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{target_metrics['sharpe_ratio']:.3f}")
                
                with col2:
                    target_pie_fig = analyzer.create_portfolio_composition_chart(
                        target_weights, data['asset_names'], "Target Return Portfolio"
                    )
                    st.plotly_chart(target_pie_fig, use_container_width=True)
    
    with tab4:
        if show_monte_carlo:
            st.markdown("## 🎲 Monte Carlo Simulation")
            
            # Simulation parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_simulations = st.selectbox("Number of Simulations", [1000, 5000, 10000], index=1)
            with col2:
                time_horizon = st.selectbox("Time Horizon (days)", [63, 126, 252], index=2)
            with col3:
                portfolio_choice = st.selectbox("Portfolio Type", ["Equal Weight", "Min Variance", "Max Sharpe"])
            
            if st.button("🚀 Run Monte Carlo Simulation"):
                # Select portfolio weights
                if portfolio_choice == "Equal Weight":
                    sim_weights = np.ones(n_assets) / n_assets
                elif portfolio_choice == "Min Variance":
                    sim_weights = analyzer.optimize_portfolio(data, 'min_variance')
                else:
                    sim_weights = analyzer.optimize_portfolio(data, 'max_sharpe')
                
                with st.spinner(f"Running {n_simulations:,} simulations..."):
                    mc_results = analyzer.monte_carlo_simulation(
                        sim_weights, data, n_simulations, time_horizon
                    )
                
                # Display results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h4>Simulation Results</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Mean Final Value", f"{mc_results['mean_final_value']:.3f}")
                    st.metric("5th Percentile", f"{mc_results['percentile_5']:.3f}")
                    st.metric("95th Percentile", f"{mc_results['percentile_95']:.3f}")
                    st.metric("Probability of Loss", f"{mc_results['probability_loss']:.1%}")
                
                with col2:
                    mc_fig = analyzer.create_monte_carlo_histogram(mc_results['final_values'])
                    st.plotly_chart(mc_fig, use_container_width=True)
    
    with tab5:
        if show_risk_analysis:
            st.markdown("## ⚠️ Risk Analysis")
            
            # Portfolio selection for risk analysis
            risk_portfolio = st.selectbox(
                "Select Portfolio for Risk Analysis",
                ["Equal Weight", "Min Variance", "Max Sharpe"],
                key="risk_portfolio"
            )
            
            # Get portfolio weights
            if risk_portfolio == "Equal Weight":
                risk_weights = np.ones(n_assets) / n_assets
            elif risk_portfolio == "Min Variance":
                risk_weights = analyzer.optimize_portfolio(data, 'min_variance')
            else:
                risk_weights = analyzer.optimize_portfolio(data, 'max_sharpe')
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(data['returns'], risk_weights)
            
            # Risk metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                var_95 = analyzer.calculate_var(portfolio_returns, 0.95)
                st.metric("VaR (95%)", f"{var_95:.4f}")
            
            with col2:
                var_99 = analyzer.calculate_var(portfolio_returns, 0.99)
                st.metric("VaR (99%)", f"{var_99:.4f}")
            
            with col3:
                # Expected Shortfall
                var_threshold = np.percentile(portfolio_returns, 5)
                tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
                expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
                st.metric("Expected Shortfall", f"{expected_shortfall:.4f}")
            
            with col4:
                # Maximum Drawdown
                cumulative_returns = np.cumprod(1 + portfolio_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdown)
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            
            # Risk decomposition
            st.markdown("### 📊 Risk Attribution")
            
            # Calculate risk contributions
            portfolio_variance = np.dot(risk_weights.T, np.dot(data['cov_matrix'], risk_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            marginal_contrib = np.dot(data['cov_matrix'], risk_weights) / portfolio_volatility
            component_contrib = risk_weights * marginal_contrib
            percent_contrib = component_contrib / portfolio_volatility * 100
            
            risk_df = pd.DataFrame({
                'Asset': data['asset_names'],
                'Weight (%)': risk_weights * 100,
                'Risk Contribution (%)': percent_contrib
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(risk_df, use_container_width=True)
            
            with col2:
                # Risk contribution pie chart
                risk_contrib_fig = analyzer.create_portfolio_composition_chart(
                    percent_contrib, data['asset_names'], "Risk Contribution"
                )
                st.plotly_chart(risk_contrib_fig, use_container_width=True)
    
    # Download section
    st.markdown("---")
    st.markdown("## 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Market Data"):
            market_df = pd.DataFrame(data['returns'], columns=data['asset_names'])
            csv = market_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Market Data (CSV)",
                data=csv,
                file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🎯 Export Portfolio Weights"):
            if 'strategy_results' in locals():
                weights_df = pd.DataFrame({
                    name: metrics['weights'] for name, metrics in strategy_results.items()
                }, index=data['asset_names'])
                csv = weights_df.to_csv()
                st.download_button(
                    label="📥 Download Weights (CSV)",
                    data=csv,
                    file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with col3:
        if st.button("🔄 Reset Analysis"):
            st.session_state.clear()
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌟 Built with Streamlit, NumPy & Plotly | Portfolio Analyzer</p>
        <p>Advanced Quantitative Finance • Real-time Optimization • Interactive Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()