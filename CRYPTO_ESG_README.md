# Professional Crypto ESG Investment Dashboard 🌱📊

A comprehensive, real-time cryptocurrency portfolio tracker with Environmental, Social, and Governance (ESG) scoring and advanced analytics.

## 🚀 Key Features

### Core Functionality
- **Real-time Price Data**: Live cryptocurrency prices from CoinGecko API
- **ESG Scoring**: Comprehensive ESG ratings based on consensus mechanisms and sustainability practices
- **Portfolio Management**: Dynamic add/remove assets with persistent session state
- **Advanced Metrics**: Weighted ESG scores, risk assessment, and diversification analysis

### Professional Visualizations
- **Interactive Charts**: Plotly-powered responsive visualizations
- **Multi-tab Analytics**: Organized dashboard with allocation, ESG analysis, performance, and historical views
- **Custom Styling**: Professional CSS styling with gradient themes and alert systems

### Advanced Analytics
- **Portfolio Metrics**:
  - Total portfolio value with 24h change tracking
  - Value-weighted ESG score calculation
  - Risk assessment based on ESG inverse scoring
  - Diversification score using Herfindahl-Hirschman Index

- **Performance Tracking**:
  - 24-hour performance monitoring
  - Market cap and volume analysis
  - Historical price tracking (7-365 days)
  - Performance leaderboard

### Smart Features
- **Auto-refresh**: Optional 30-second auto-refresh capability
- **Progress Indicators**: Real-time data fetching progress bars
- **Error Handling**: Robust API error handling with fallback data
- **Responsive Design**: Mobile-friendly layout with wide screen optimization

## 📊 Dashboard Sections

### 1. Portfolio Overview
- Comprehensive asset table with formatting and color-coding
- Real-time price updates and 24h change indicators
- Market cap and trading volume information

### 2. Key Metrics Row
- **Portfolio Value**: Total value with daily change tracking
- **Weighted ESG Score**: Value-weighted sustainability rating
- **Risk Score**: Investment risk assessment
- **Diversification Score**: Portfolio concentration analysis

### 3. Analytics Tabs

#### 💰 Allocation Tab
- Interactive pie chart for portfolio allocation
- ESG vs. Value scatter plot with bubble sizing
- Technology category breakdown

#### 🌱 ESG Analysis Tab
- Comprehensive 4-panel ESG visualization:
  - ESG scores by asset (bar chart)
  - Value vs. ESG correlation (scatter plot)
  - Risk category distribution (pie chart)
  - ESG score distribution (histogram)
- Smart ESG improvement recommendations
- Low-ESG asset warnings

#### 📊 Performance Tab
- 24-hour performance bar chart with color coding
- Performance leaderboard with ranking
- Volume and market cap analysis

#### 📈 Historical Tab
- Multi-asset historical price tracking
- Customizable time periods (7-365 days)
- Weighted historical performance by holdings

## 🌟 Advanced Features

### ESG Scoring System
The dashboard includes a comprehensive ESG database with scores for major cryptocurrencies:

- **High ESG (80+)**: Algorand (92), Cardano (88), Tezos (85), Polkadot (82)
- **Medium ESG (60-79)**: Ethereum (78), Chainlink (75), Avalanche (72), Solana (70)
- **Low ESG (<60)**: Bitcoin (35) - due to Proof of Work energy consumption

### Risk Categories
- **Low Risk**: Algorand, Cardano, Tezos
- **Medium Risk**: Ethereum, Polkadot, Chainlink, Solana, Avalanche, Cosmos
- **High Risk**: Bitcoin

### Technology Categories
- **POS**: Proof of Stake cryptocurrencies (most sustainable)
- **POW**: Proof of Work cryptocurrencies (high energy consumption)
- **Oracle**: Oracle network tokens
- **Layer 1**: Base layer blockchain protocols

## 🔧 Technical Implementation

### Architecture
- **Object-Oriented Design**: Clean class-based architecture with separation of concerns
- **Session Management**: Streamlit session state for persistent portfolio data
- **API Integration**: Robust CoinGecko API integration with error handling
- **Data Processing**: Pandas-based data manipulation and analysis

### Performance Optimizations
- **Efficient API Calls**: Batch API requests where possible
- **Caching Strategy**: Session-based caching for reduced API calls
- **Progress Feedback**: Real-time progress indicators for better UX
- **Error Recovery**: Graceful degradation when API calls fail

### Visualization Stack
- **Plotly**: Interactive charts with hover effects and responsiveness
- **Matplotlib/Seaborn**: Statistical visualizations
- **Custom CSS**: Professional styling and responsive design
- **Color Schemes**: Semantic color coding (green=good, red=bad, yellow=neutral)

## 🚀 Getting Started

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run crypto_esg_dashboard.py
```

### Usage
1. **Add Assets**: Use the sidebar to add cryptocurrencies to your portfolio
2. **Set Amounts**: Specify the quantity of each asset you hold
3. **Monitor Performance**: Track real-time portfolio value and performance
4. **Analyze ESG**: Review sustainability scores and get improvement recommendations
5. **View Trends**: Explore historical performance and market data

### Configuration Options
- **Auto-refresh**: Enable 30-second automatic data updates
- **Historical Data**: Toggle historical chart display
- **Time Period**: Adjust historical analysis period (7-365 days)

## 📊 Data Sources

- **Price Data**: CoinGecko API (real-time prices, market data, historical charts)
- **ESG Scores**: Curated database based on consensus mechanisms, energy efficiency, and sustainability practices
- **Market Data**: Market capitalization, trading volume, 24h price changes

## 🎯 Future Enhancements

Potential improvements for the dashboard:
- Integration with portfolio tracking APIs (Coinbase, Binance)
- Advanced technical analysis indicators
- Portfolio optimization suggestions
- Social sentiment analysis
- News feed integration
- Export functionality (PDF reports, CSV data)
- Multi-currency support
- Mobile app version

## 📝 Notes

- ESG scores are estimated based on publicly available information about consensus mechanisms and sustainability initiatives
- Real-time data depends on CoinGecko API availability
- Portfolio data is stored in session state (resets on browser refresh)
- The dashboard is designed for educational and analytical purposes

## 🔐 Security Considerations

- No API keys required for basic functionality
- No personal financial data stored permanently
- All data processing happens locally in the browser session
- No external data transmission beyond public API calls
