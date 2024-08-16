import yfinance as yf

# Download historical data for Reliance Industries Ltd on NSE
data = yf.download("RELIANCE.NS", start="2023-01-01", end="2023-12-31")

# Save to CSV
data.to_csv("prices.csv")
