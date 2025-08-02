import requests
from bs4 import BeautifulSoup

def fetch_optioncharts_overview():
    url = "https://chartexchange.com/symbol/nyse-gme/optionchain/summary/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    iv = None
    ratio = None

    try:
        # Parse implied volatility from the summary table
        iv_cell = soup.find("td", string=lambda s: s and "Implied Volatility" in s)
        if iv_cell:
            iv_value_cell = iv_cell.find_next("td")
            if iv_value_cell:
                iv_text = iv_value_cell.text.strip()
                iv = float(iv_text.replace("%", "").strip())
        
        # Parse Put/Call Ratio from the summary table
        ratio_cell = soup.find("td", string=lambda s: s and "Put/Call OI Ratio" in s)
        if ratio_cell:
            ratio_value_cell = ratio_cell.find_next("td")
            if ratio_value_cell:
                ratio = ratio_value_cell.text.strip()

    except Exception as e:
        print("⚠️ Error parsing OptionCharts summary:", e)

    return iv, ratio

