import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Professional Crypto ESG Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .success-alert {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .warning-alert {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    .danger-alert {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CryptoESGDashboard:
    def __init__(self):
        self.api_base = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoESGDashboard/1.0'
        })
        
        # Extended cryptocurrency data with comprehensive ESG scores
        self.crypto_database = {
            "bitcoin": {"name": "Bitcoin", "esg": 35, "category": "POW", "risk": "High"},
            "ethereum": {"name": "Ethereum", "esg": 78, "category": "POS", "risk": "Medium"},
            "cardano": {"name": "Cardano", "esg": 88, "category": "POS", "risk": "Low"},
            "polkadot": {"name": "Polkadot", "esg": 82, "category": "POS", "risk": "Medium"},
            "chainlink": {"name": "Chainlink", "esg": 75, "category": "Oracle", "risk": "Medium"},
            "solana": {"name": "Solana", "esg": 70, "category": "POS", "risk": "Medium"},
            "avalanche-2": {"name": "Avalanche", "esg": 72, "category": "POS", "risk": "Medium"},
            "algorand": {"name": "Algorand", "esg": 92, "category": "POS", "risk": "Low"},
            "tezos": {"name": "Tezos", "esg": 85, "category": "POS", "risk": "Low"},
            "cosmos": {"name": "Cosmos", "esg": 80, "category": "POS", "risk": "Medium"},
        }
    
    def get_price_data(self, symbol: str) -> Dict:
        """Get current price and 24h change for a cryptocurrency"""
        try:
            url = f"{self.api_base}/simple/price"
            params = {
                'ids': symbol,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if symbol in data:
                return {
                    'price': data[symbol]['usd'],
                    'change_24h': data[symbol].get('usd_24h_change', 0),
                    'market_cap': data[symbol].get('usd_market_cap', 0),
                    'volume_24h': data[symbol].get('usd_24h_vol', 0)
                }
            else:
                return {'price': 0, 'change_24h': 0, 'market_cap': 0, 'volume_24h': 0}
        
        except Exception as e:
            st.warning(f"Error fetching data for {symbol}: {str(e)}")
            return {'price': 0, 'change_24h': 0, 'market_cap': 0, 'volume_24h': 0}
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data"""
        try:
            url = f"{self.api_base}/coins/{symbol}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'prices' in data:
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.drop('timestamp', axis=1)
                return df
            else:
                return pd.DataFrame()
        
        except Exception as e:
            st.warning(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, portfolio_data: List[Dict]) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        total_value = sum(item['value'] for item in portfolio_data)
        
        if total_value == 0:
            return {
                'total_value': 0,
                'weighted_esg': 0,
                'daily_change': 0,
                'daily_change_pct': 0,
                'risk_score': 0,
                'diversification_score': 0
            }
        
        # Weighted ESG Score
        weighted_esg = sum(item['value'] * item['esg'] for item in portfolio_data) / total_value
        
        # Daily change
        daily_change = sum(item['amount'] * item['price'] * (item['change_24h'] / 100) 
                          for item in portfolio_data)
        daily_change_pct = (daily_change / total_value) * 100 if total_value > 0 else 0
        
        # Risk Score (inverse of ESG, weighted by value)
        risk_score = 100 - weighted_esg
        
        # Diversification Score (based on number of assets and distribution)
        n_assets = len(portfolio_data)
        if n_assets > 1:
            values = [item['value'] for item in portfolio_data]
            weights = np.array(values) / total_value
            hhi = sum(w**2 for w in weights)  # Herfindahl-Hirschman Index
            diversification_score = (1 - hhi) * 100
        else:
            diversification_score = 0
        
        return {
            'total_value': total_value,
            'weighted_esg': weighted_esg,
            'daily_change': daily_change,
            'daily_change_pct': daily_change_pct,
            'risk_score': risk_score,
            'diversification_score': diversification_score
        }
    
    def create_portfolio_charts(self, df: pd.DataFrame) -> Tuple:
        """Create advanced portfolio visualization charts"""
        
        # 1. Portfolio Allocation Pie Chart
        fig_pie = px.pie(
            df, 
            values='Value (USD)', 
            names='Name',
            title="Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )
        fig_pie.update_layout(showlegend=True, height=500)
        
        # 2. ESG vs Value Scatter Plot
        fig_scatter = px.scatter(
            df,
            x='ESG Score',
            y='Value (USD)',
            size='Amount',
            color='24h Change (%)',
            hover_name='Name',
            title="ESG Score vs Portfolio Value",
            color_continuous_scale='RdYlGn'
        )
        fig_scatter.update_layout(height=500)
        
        # 3. Performance Bar Chart
        fig_bar = px.bar(
            df,
            x='Name',
            y='24h Change (%)',
            color='24h Change (%)',
            title="24-Hour Performance",
            color_continuous_scale='RdYlGn'
        )
        fig_bar.update_layout(height=400)
        
        return fig_pie, fig_scatter, fig_bar
    
    def create_esg_analysis_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive ESG analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ESG Scores by Asset', 'Value vs ESG Score', 
                           'Risk Categories', 'ESG Score Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # ESG Scores Bar Chart
        fig.add_trace(
            go.Bar(x=df['Name'], y=df['ESG Score'], name='ESG Score',
                   marker_color=df['ESG Score'], colorscale='RdYlGn'),
            row=1, col=1
        )
        
        # Value vs ESG Scatter
        fig.add_trace(
            go.Scatter(x=df['ESG Score'], y=df['Value (USD)'], 
                      mode='markers+text', name='Assets',
                      text=df['Name'], textposition="top center",
                      marker=dict(size=10, colorscale='RdYlGn', 
                                color=df['ESG Score'])),
            row=1, col=2
        )
        
        # Risk Categories Pie
        risk_counts = df['Risk Category'].value_counts()
        fig.add_trace(
            go.Pie(labels=risk_counts.index, values=risk_counts.values,
                   name='Risk Categories'),
            row=2, col=1
        )
        
        # ESG Distribution Histogram
        fig.add_trace(
            go.Histogram(x=df['ESG Score'], name='ESG Distribution',
                        nbinsx=10, marker_color='lightblue'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        return fig

def main():
    dashboard = CryptoESGDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🌱 Professional Crypto ESG Investment Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Portfolio Configuration</div>', 
                       unsafe_allow_html=True)
    
    # Portfolio Management
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = [
            {"name": "Bitcoin", "symbol": "bitcoin", "amount": 0.5},
            {"name": "Ethereum", "symbol": "ethereum", "amount": 1.2},
            {"name": "Cardano", "symbol": "cardano", "amount": 100},
        ]
    
    # Add/Remove Assets
    st.sidebar.subheader("📝 Manage Portfolio")
    
    # Add new asset
    available_cryptos = list(dashboard.crypto_database.keys())
    selected_crypto = st.sidebar.selectbox("Add Cryptocurrency:", available_cryptos)
    amount = st.sidebar.number_input("Amount:", min_value=0.0, value=1.0, step=0.1)
    
    if st.sidebar.button("➕ Add Asset"):
        crypto_info = dashboard.crypto_database[selected_crypto]
        new_asset = {
            "name": crypto_info["name"],
            "symbol": selected_crypto,
            "amount": amount
        }
        st.session_state.portfolio.append(new_asset)
        st.sidebar.success(f"Added {crypto_info['name']} to portfolio!")
    
    # Remove asset
    if st.session_state.portfolio:
        portfolio_names = [asset["name"] for asset in st.session_state.portfolio]
        asset_to_remove = st.sidebar.selectbox("Remove Asset:", ["None"] + portfolio_names)
        
        if st.sidebar.button("🗑️ Remove Asset") and asset_to_remove != "None":
            st.session_state.portfolio = [
                asset for asset in st.session_state.portfolio 
                if asset["name"] != asset_to_remove
            ]
            st.sidebar.success(f"Removed {asset_to_remove} from portfolio!")
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    show_historical = st.sidebar.checkbox("📈 Show Historical Data", value=True)
    historical_days = st.sidebar.slider("Historical Period (days)", 7, 365, 30)
    
    # Main Dashboard
    if not st.session_state.portfolio:
        st.warning("⚠️ No assets in portfolio. Please add some cryptocurrencies from the sidebar.")
        return
    
    # Data Collection with Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    portfolio_data = []
    total_assets = len(st.session_state.portfolio)
    
    for i, asset in enumerate(st.session_state.portfolio):
        status_text.text(f"Fetching data for {asset['name']}...")
        progress_bar.progress((i + 1) / total_assets)
        
        price_data = dashboard.get_price_data(asset["symbol"])
        crypto_info = dashboard.crypto_database.get(asset["symbol"], {})
        
        portfolio_data.append({
            "name": asset["name"],
            "symbol": asset["symbol"],
            "amount": asset["amount"],
            "price": price_data["price"],
            "value": asset["amount"] * price_data["price"],
            "change_24h": price_data["change_24h"],
            "market_cap": price_data["market_cap"],
            "volume_24h": price_data["volume_24h"],
            "esg": crypto_info.get("esg", 50),
            "category": crypto_info.get("category", "Unknown"),
            "risk": crypto_info.get("risk", "Medium")
        })
    
    progress_bar.empty()
    status_text.empty()
    
    # Calculate Metrics
    metrics = dashboard.calculate_portfolio_metrics(portfolio_data)
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            "Name": item["name"],
            "Amount": item["amount"],
            "Price (USD)": item["price"],
            "Value (USD)": item["value"],
            "24h Change (%)": item["change_24h"],
            "Market Cap": item["market_cap"],
            "Volume 24h": item["volume_24h"],
            "ESG Score": item["esg"],
            "Category": item["category"],
            "Risk Category": item["risk"]
        }
        for item in portfolio_data
    ])
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        change_color = "🟢" if metrics['daily_change'] >= 0 else "🔴"
        st.metric(
            "💰 Portfolio Value",
            f"${metrics['total_value']:,.2f}",
            f"{change_color} ${metrics['daily_change']:,.2f} ({metrics['daily_change_pct']:+.2f}%)"
        )
    
    with col2:
        esg_color = "🟢" if metrics['weighted_esg'] >= 70 else "🟡" if metrics['weighted_esg'] >= 50 else "🔴"
        st.metric(
            "🌱 Weighted ESG Score",
            f"{esg_color} {metrics['weighted_esg']:.1f}/100"
        )
    
    with col3:
        risk_color = "🟢" if metrics['risk_score'] <= 30 else "🟡" if metrics['risk_score'] <= 60 else "🔴"
        st.metric(
            "⚠️ Risk Score",
            f"{risk_color} {metrics['risk_score']:.1f}/100"
        )
    
    with col4:
        div_color = "🟢" if metrics['diversification_score'] >= 60 else "🟡" if metrics['diversification_score'] >= 30 else "🔴"
        st.metric(
            "📊 Diversification",
            f"{div_color} {metrics['diversification_score']:.1f}/100"
        )
    
    # ESG Alerts
    if metrics['weighted_esg'] < 50:
        st.markdown(
            '<div class="alert-box danger-alert">⚠️ <strong>Low ESG Alert:</strong> Your portfolio has a low ESG score. Consider adding more sustainable cryptocurrencies.</div>',
            unsafe_allow_html=True
        )
    elif metrics['weighted_esg'] >= 80:
        st.markdown(
            '<div class="alert-box success-alert">✅ <strong>Excellent ESG Score:</strong> Your portfolio demonstrates strong sustainability commitment!</div>',
            unsafe_allow_html=True
        )
    
    # Portfolio Table
    st.subheader("📊 Portfolio Overview")
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df = display_df.style.format({
        "Price (USD)": "${:,.4f}",
        "Value (USD)": "${:,.2f}",
        "24h Change (%)": "{:+.2f}%",
        "Market Cap": "${:,.0f}",
        "Volume 24h": "${:,.0f}"
    }).background_gradient(subset=['ESG Score'], cmap='RdYlGn', vmin=0, vmax=100)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Charts Section
    st.subheader("📈 Portfolio Analytics")
    
    # Create charts
    fig_pie, fig_scatter, fig_bar = dashboard.create_portfolio_charts(df)
    
    # Display charts in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Allocation", "🌱 ESG Analysis", "📊 Performance", "📈 Historical"])
    
    with tab1:
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Additional allocation insights
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col2:
            # Category breakdown
            category_dist = df.groupby('Category')['Value (USD)'].sum()
            fig_category = px.bar(
                x=category_dist.index, 
                y=category_dist.values,
                title="Value by Technology Category",
                labels={'y': 'Value (USD)', 'x': 'Category'}
            )
            st.plotly_chart(fig_category, use_container_width=True)
    
    with tab2:
        esg_fig = dashboard.create_esg_analysis_chart(df)
        st.plotly_chart(esg_fig, use_container_width=True)
        
        # ESG Recommendations
        st.subheader("🎯 ESG Improvement Recommendations")
        low_esg_assets = df[df['ESG Score'] < 60]
        if not low_esg_assets.empty:
            st.warning("Consider reducing exposure to these low-ESG assets:")
            for _, asset in low_esg_assets.iterrows():
                st.write(f"• {asset['Name']}: ESG Score {asset['ESG Score']}/100")
        
        high_esg_suggestions = ["Algorand (92)", "Cardano (88)", "Tezos (85)", "Polkadot (82)"]
        st.success("Consider adding these high-ESG cryptocurrencies:")
        for suggestion in high_esg_suggestions:
            st.write(f"• {suggestion}")
    
    with tab3:
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Performance metrics table
        perf_df = df[['Name', '24h Change (%)', 'Volume 24h', 'Market Cap']].sort_values('24h Change (%)', ascending=False)
        st.subheader("🏆 Performance Leaderboard")
        st.dataframe(
            perf_df.style.format({
                "24h Change (%)": "{:+.2f}%",
                "Volume 24h": "${:,.0f}",
                "Market Cap": "${:,.0f}"
            }).background_gradient(subset=['24h Change (%)'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with tab4:
        if show_historical:
            st.subheader(f"📈 Historical Performance ({historical_days} days)")
            
            # Fetch historical data for portfolio assets
            historical_data = {}
            progress_hist = st.progress(0)
            
            for i, asset in enumerate(portfolio_data):
                progress_hist.progress((i + 1) / len(portfolio_data))
                hist_df = dashboard.get_historical_data(asset['symbol'], historical_days)
                if not hist_df.empty:
                    hist_df['weighted_price'] = hist_df['price'] * asset['amount']
                    historical_data[asset['name']] = hist_df
            
            progress_hist.empty()
            
            if historical_data:
                # Create combined historical chart
                fig_hist = go.Figure()
                
                for name, hist_df in historical_data.items():
                    fig_hist.add_trace(
                        go.Scatter(
                            x=hist_df['date'],
                            y=hist_df['weighted_price'],
                            mode='lines',
                            name=name,
                            hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>'
                        )
                    )
                
                fig_hist.update_layout(
                    title="Portfolio Assets Historical Performance (Weighted by Holdings)",
                    xaxis_title="Date",
                    yaxis_title="Value (USD)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Enable 'Show Historical Data' in the sidebar to view historical performance.")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>Professional Crypto ESG Dashboard</strong> | 
            Data provided by CoinGecko API | 
            ESG scores are estimated based on consensus mechanisms and sustainability practices</p>
            <p><em>Last updated: {}</em></p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
