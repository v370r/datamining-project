from sec_edgar_downloader import Downloader

# Download filings to the current working directory
dl = Downloader("StockLens", "pml9652@gmail.com", "./data")

# Get all 10-K filings for Microsoft without the filing details
# dl.get("10-K", "MSFT", download_details=False)
dl.get("10-K", "NVDA", download_details=False, limit=5)