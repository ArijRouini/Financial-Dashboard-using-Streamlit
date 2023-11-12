# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:34:11 2023

@author: Arij Rouini
"""

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret



# ***************************************************       ***********************************************************************************************************************************************************************************************************************************************************
#                                                    Streamlit 
#****************************************************        ************************************************



# Libraries
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import requests


# ***************************************************       ***********************************************************************************************************************************************************************************************************************************************************
#                                                    ♦ TAB1 ♦
#****************************************************      *****************************************************************************************************************************************************************************************************




# Function to get S&P 500 tickers
def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return table[0]['Symbol']

# Function to calculate start date based on selected duration
def calculate_start_end_dates(end_date, duration):
    if duration == '1D':
        return end_date - timedelta(days=1)
    elif duration == '5D':
        return end_date - timedelta(days=5)
    elif duration == '1M':
        return end_date - timedelta(days=30)
    elif duration == '6M':
        return end_date - timedelta(days=182)
    elif duration == 'YTD':
        return datetime(end_date.year, 1, 1)
    elif duration == '1Y':
        return end_date - timedelta(days=365)
    elif duration == '3Y':
        return end_date - timedelta(days=3*365)
    elif duration == '5Y':
        return end_date - timedelta(days=5*365)
    elif duration == 'Max':
        return datetime(1970, 1, 1)

# Function to get additional information from Yahoo Finance
def get_company_info(ticker):
    try:
        return YFinance(ticker).info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Function to render the Sidebar for Summary Tab
def render_summary_sidebar():
    st.sidebar.header("Select Options")
    
    # To select Ticker
    ticker = st.sidebar.selectbox("Select Ticker", get_sp500_tickers())
    
    # To select Time Duration
    durations = ['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'Max']
    duration = st.sidebar.selectbox("Select Time Duration", durations)
    
    # to calculate Start Date
    end_date = st.sidebar.date_input("End Date", datetime(2023, 11, 8))
    start_date = calculate_start_end_dates(end_date, duration)
    
    # Start Date Input
    start_date = st.sidebar.date_input("Start Date", start_date)

    # To display the Start Date and End Date in columns
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.write(f"Selected Start Date: {start_date}")
    with col2:
        st.write(f"Selected End Date: {end_date}")                           #

    # Inform the user to click "Update Data" for changes with styling
    st.sidebar.markdown("<p style='color: #9932CC;'> Please click on 'Update Data' to apply or modify selections </p>", unsafe_allow_html=True)
    
    # Update Data button
    update_button = st.sidebar.button("Update Data")
    
    return ticker, start_date, end_date, duration, update_button

# Function to display the Summary Tab
def tab_summary():
    ticker, start_date, end_date, duration, update_button = render_summary_sidebar()
    st.subheader("Stock price and volume change over time")
    
    # Check if the "Update Data" button is clicked
    if update_button:
        # Fetch data for the selected ticker and time period
        if duration == '1D':
            stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1m')
        else:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Display the selected start and end dates
        st.write(f"Selected Start Date: {start_date}")
        st.write(f"Selected End Date: {end_date}")

        # Create an interactive line chart for the stock data
        fig = go.Figure()

        # Line for the stock price changes
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Stock Price', line=dict(color='black')))
        
        # Add a filled area below the line chart for stock price
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], fill='tozeroy', fillcolor='rgba(128, 128, 128, 0.3)', line=dict(color='rgba(0,0,0,0)')))
        
        # Add a bar chart for the volume inside the gray area
        volume_scaling_factor = 0.000001  
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'] * volume_scaling_factor, name='Volume', marker_color="#9370db", yaxis='y2'))

        # Update layout for better visualization
        fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=True,
                          hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                          yaxis=dict(title='Stock Price (USD)'),
                          yaxis2=dict(title='Volume', overlaying='y', side='right'))


        # Enhance interactivity with more informative tooltips
        fig.update_traces(
            hoverinfo='x+y',
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d} <br>' +
                          '<b>Time:</b> %{x|%I:%M %p} <br>' +
                          '<b>Stock Price:</b> $%{y:.2f}' +
                          '<extra><b>Open:</b> %{customdata[0]:.2f} <br>' +
                          '<b>Low:</b> %{customdata[2]:.2f} <br>' +
                          '<b>Volume:</b> %{customdata[3]:,}</extra>',
            customdata=np.column_stack((stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Volume'])),
        )

        # Display the interactive chart
        st.plotly_chart(fig)
        
        
        # Get company information for the selected ticker 
        company_info = get_company_info(ticker)

        if company_info:
            # Show the company description using markdown + HTML
            st.subheader("Business Summary")
            st.markdown('<div style="text-align: justify;">' + \
                        company_info.get('longBusinessSummary', 'No business summary available') + \
                        '</div><br>',
                        unsafe_allow_html=True)
                
            # Get company information for the selected ticker
            stakeholders_info = get_company_info(ticker)
            
            if stakeholders_info:
                # Show information about stakeholders
                st.subheader("Major Shareholders:")
            
                # Get the company officers
                company_officers = stakeholders_info.get("companyOfficers", [])
            
                # Create a DataFrame with only the desired columns
                desired_columns = ['name', 'title','age']
                company_officers_df = pd.DataFrame(company_officers)[desired_columns]
            
                # Display the company officers' information
                st.write(company_officers_df)        
            
            # Show some statistics
            
            st.subheader("Key Statistics")
            
            info_keys = {
                'Previous Close': company_info.get('previousClose', 'N/A'),
                'Open': company_info.get('open', 'N/A'),
                'Bid': company_info.get('bid', 'N/A'),
                'Ask': company_info.get('ask', 'N/A'),
                'Market Cap': company_info.get('marketCap', 'N/A'),
                'Volume': company_info.get('volume', 'N/A'),
                'Day\'s Range': f"{company_info.get('dayLow', 'N/A')} - {company_info.get('dayHigh', 'N/A')}",
                '52 Week Range': f"{company_info.get('fiftyTwoWeekLow', 'N/A')} - {company_info.get('fiftyTwoWeekHigh', 'N/A')}",
                'Avg. Volume': company_info.get('averageVolume', 'N/A'),
                'Beta (5Y Monthly)': company_info.get('beta', 'N/A'),
                'PE Ratio (TTM)': company_info.get('trailingPE', 'N/A'),
                'EPS (TTM)': company_info.get('trailingEps', 'N/A'),
                'Earnings Date': ', '.join(company_info.get('earningsDates', ['N/A'])),
                'Forward Dividend & Yield': company_info.get('dividendYield', 'N/A'),
                '1y Target Est.': company_info.get('targetMeanPrice', 'N/A')
            }


            # Convert to DataFrame
            company_stats = pd.DataFrame({'Value': pd.Series(info_keys)})
            st.write(company_stats)

        

# ***************************************************       ***********************************************************************************************************************************************************************************************************************************************************
#                                                    ♦ TAB2 ♦
#****************************************************      ***********************************************************************************************************************************************************************************************************************************************************


# Function to render the Sidebar for Chart Tab
def render_chart_sidebar():
    st.sidebar.header("Select Options")

    # Select Ticker
    ticker = st.sidebar.selectbox("Select Ticker", get_sp500_tickers())

    # Select Time Duration
    durations = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'Max']
    duration = st.sidebar.selectbox("Select Time Duration", durations)

    # Select Time Interval
    time_intervals = {'Day': '1d', 'Month': '1mo'}
    time_interval = time_intervals[st.sidebar.selectbox("Select Time Interval", list(time_intervals.keys()))]

    # Date Inputs for Start and End Dates
    end_date = st.sidebar.date_input("End Date", datetime.now())
    start_date = st.sidebar.date_input("Start Date", calculate_start_end_dates(end_date, duration))

    # Select Chart Type
    chart_types = ['Line Plot', 'Candle Plot']
    chart_type = st.sidebar.selectbox("Select Chart Type", chart_types)

    # Update Data button
    update_button = st.sidebar.button("Update Chart")

    return ticker, start_date, end_date, duration, time_interval, chart_type, update_button



# Function to display the Chart Tab
def tab_chart():
    ticker, start_date, end_date, duration, time_interval, chart_type, update_button = render_chart_sidebar()

    # Check if the "Update Chart" button is clicked
    if update_button:
        # looking for data for the selected ticker and time period
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=time_interval)

        # Create an interactive chart
        fig = go.Figure()

        # Check chart type
        if chart_type == 'Line Plot':
            # Add line plot for stock price
            if duration == '1D':
                stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1m')
            else:
                stock_data = yf.download(ticker, start=start_date, end=end_date)

            # Display the selected start and end dates
            st.write(f"Selected Start Date: {start_date}")
            st.write(f"Selected End Date: {end_date}")

            # Create an interactive wavy line chart for the stock data
            fig = go.Figure()

            # Add a wavy line for the stock price
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Stock Price',
                                     line=dict(color='black')))

            # Add a filled area below the line chart for stock price
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], fill='tozeroy',
                                     fillcolor='rgba(128, 128, 128, 0.3)', line=dict(color='rgba(0,0,0,0)')))

            # Add a bar chart for the volume inside the gray area
            volume_scaling_factor = 0.000001  # Adjust this factor to control the size of the volume bars
            fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'] * volume_scaling_factor, name='Volume',
                                 marker_color='purple', yaxis='y2'))

            # Update layout for better visualization
            fig.update_layout(showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=True,
                              hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                              yaxis=dict(title='Stock Price (USD)'),
                              yaxis2=dict(title='Volume', overlaying='y', side='right'))
        elif chart_type == 'Candle Plot':
            # Create a candlestick chart for the stock data
            fig.add_trace(go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'],
                                         name='Candlestick'))

            # Add a bar chart for volume below the candlestick chart
            fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', marker_color='gray'))

            # Add a line for the moving average (MA)
            ma_window = 50
            stock_data['MA'] = stock_data['Close'].rolling(window=ma_window).mean()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA'], mode='lines',
                                     name=f'MA-{ma_window}', line=dict(color='blue')))

            # Update layout for better visualization
            title_text = f"{ticker} Stock Price Over Time - Please Click on 'Volume' in the legend to see the variation of the stock in candlestick form"
            fig.update_layout(title=title_text,
                              xaxis_title='Date and Time',
                              yaxis_title='Stock Price (USD)',
                              xaxis_rangeslider_visible=True)

        # Display the interactive chart
        st.plotly_chart(fig)



# ***************************************************      ***********************************************************************************************************************************************************************************************************************************************************
#                                                   ♦ TAB 3 ♦
#****************************************************      ***********************************************************************************************************************************************************************************************************************************************************



# Function to render the Sidebar for Financials Tab
def render_financials_sidebar():
    st.sidebar.header("Financials Options")

    # Select Ticker for Financials
    financials_ticker = st.sidebar.selectbox("Select Ticker", get_sp500_tickers())

    # Select Financial Statement Type
    financial_statement_types = ['Income Statement', 'Balance Sheet', 'Cash Flow']
    financial_statement_type = st.sidebar.selectbox("Select Financial Statement Type", financial_statement_types)

    # Select Period (Annual or Quarterly)
    periods = ['Annual', 'Quarterly']
    period = st.sidebar.selectbox("Select Period", periods)

    # Update Data button for Financials
    update_financials_button = st.sidebar.button("Update Financials")

    return financials_ticker, financial_statement_type, period, update_financials_button

# Function to display the Financials Tab
def tab_financials():
    financials_ticker, financial_statement_type, period, update_financials_button = render_financials_sidebar()

    # Check if the "Update Financials" button is clicked
    if update_financials_button:
        try:
            # To look for each financial statements annually and quarterly
            if period == 'Quarterly':
                if financial_statement_type == 'Income Statement':
                    financial_data = yf.Ticker(financials_ticker).quarterly_financials
                elif financial_statement_type == 'Balance Sheet':
                    financial_data = yf.Ticker(financials_ticker).quarterly_balance_sheet
                elif financial_statement_type == 'Cash Flow':
                    financial_data = yf.Ticker(financials_ticker).quarterly_cashflow
            elif period == 'Annual':
                if financial_statement_type == 'Income Statement':
                    financial_data = yf.Ticker(financials_ticker).financials
                elif financial_statement_type == 'Balance Sheet':
                    financial_data = yf.Ticker(financials_ticker).balance_sheet
                elif financial_statement_type == 'Cash Flow':
                    financial_data = yf.Ticker(financials_ticker).cashflow

            # Display the selected financial statement
            st.write(f"{financial_statement_type} for {financials_ticker}")

            # Modify the column names to include only the date part without the time component
            if 'Date' in financial_data.columns:
                financial_data['Date'] = pd.to_datetime(financial_data['Date']).dt.strftime('%y-%m-%d')
                
            st.write(financial_data)

        except yf.utils.exceptions.YFinanceError as e:
            st.error(f"An error occurred: {str(e)}")
            
            
# ***************************************************      ***********************************************************************************************************************************************************************************************************************************************************
#                                                   ♦ TAB 4 ♦
#****************************************************      ***********************************************************************************************************************************************************************************************************************************************************


# Function to render the Monte Carlo Simulation sidebar
def render_mc_simulation_sidebar():
    st.sidebar.header("Monte Carlo Simulation Options")

    # Select Ticker for Financials
    financials_ticker = st.sidebar.selectbox("Select Ticker", get_sp500_tickers())

    # Select Number of Simulations
    num_simulations_list = [200, 500, 1000]
    num_simulations = st.sidebar.selectbox("Select Number of Simulations", num_simulations_list)

    # Select Time Horizon
    time_horizon_list = [30, 60, 90]
    time_horizon = st.sidebar.selectbox("Select Time Horizon (days)", time_horizon_list)

    # Update Monte Carlo button
    update_mc_button = st.sidebar.button("Run Monte Carlo Simulation")

    return financials_ticker, num_simulations, time_horizon, update_mc_button


# Function to display the Monte Carlo Simulation Tab
def tab_mc_simulation():
    mc_ticker, num_simulations, time_horizon, update_mc_button = render_mc_simulation_sidebar()

    # Check if the "Run Monte Carlo Simulation" button is clicked
    if update_mc_button:
        try:
            # Fetch historical stock data
            stock_data = yf.Ticker(mc_ticker).history(period="1y")
            closing_prices = stock_data['Close']

            # Calculate daily returns
            daily_returns = closing_prices.pct_change().dropna()

            # Calculate mean and standard deviation of daily returns
            mu = daily_returns.mean()
            sigma = daily_returns.std()

            # Set random seed for reproducibility
            np.random.seed(42)

            # Generate Monte Carlo simulations
            simulations = np.random.normal(mu, sigma, (time_horizon, num_simulations))

            # Calculate future stock prices using geometric Brownian motion
            simulated_prices = closing_prices.iloc[-1] * np.exp(np.cumsum(simulations, axis=0))

            # Calculate VaR at 95% confidence interval
            var_95 = np.percentile(simulated_prices, 5, axis=1)

            # Create an array of days for the x-axis
            days_array = np.arange(1, time_horizon + 1)

            # Plot the simulation results using plotly
            fig = go.Figure()

            # Plot the actual closing prices
            fig.add_trace(go.Scatter(x=days_array, y=closing_prices.values, mode='lines', name='Actual Closing Prices', line=dict(color='purple', width=2)))

            # Plot the Monte Carlo simulations without legend
            colors = ['rgba(255,0,0,0.1)', 'rgba(0,255,0,0.1)', 'rgba(0,0,255,0.1)']
            for i in range(num_simulations):
                fig.add_trace(go.Scatter(x=days_array, y=simulated_prices[:, i], mode='lines', showlegend=False, line=dict(color=colors[i % len(colors)], width=1)))

            # Set layout
            fig.update_layout(title=f"Monte Carlo Simulation Results for {mc_ticker}",
                              xaxis_title="Days",
                              yaxis_title="Stock Price",
                              showlegend=True,
                              template="plotly_white")

            # Adjust y-axis scale based on actual and simulated prices
            y_range = [min(closing_prices.min(), simulated_prices.min()),
                       max(closing_prices.max(), simulated_prices.max())]
            fig.update_layout(yaxis=dict(range=y_range))

            # Display the plot
            st.plotly_chart(fig)

            # Display VaR at 95% confidence interval
            st.write(f"Estimated VaR (Value at Risk) at 95% confidence interval: [{min(var_95)}, {max(var_95)}]")

        except yf.YFinanceError as e:
            st.error(f"An error occurred: {str(e)}")



# ***************************************************      ***********************************************************************************************************************************************************************************************************************************************************
#                                                   ♦ TAB 5 ♦
#****************************************************      ***********************************************************************************************************************************************************************************************************************************************************


# Function for the stock sentiment analysis tab
def tab_stock_sentiment():
   

    # Get user input for the stock ticker
    ticker = st.sidebar.selectbox("Select Ticker", get_sp500_tickers())

    if st.button("Analyze Sentiment"):
        # Display sentiment distribution graph using Plotly
        recent_news = get_stock_news(ticker)
        plot_sentiment_distribution(recent_news)

        # Display recent news headlines with legend
        st.markdown('<p style="color:purple; font-weight:bold;">Recent News Headlines:</p>', unsafe_allow_html=True)
        for headline, sentiment_color in recent_news:
            st.markdown(f"<font style='font-size:16px' color='{sentiment_color}'>&bull;</font> {headline}", unsafe_allow_html=True)

        # Legend for headlines
        st.markdown('<p style="color:green; font-weight:bold;">Legend:</p>', unsafe_allow_html=True)
        st.markdown("<font color='green'>&bull;</font> Positive &emsp; <font color='red'>&bull;</font> Negative &emsp; <font color='gray'>&bull;</font> Neutral", unsafe_allow_html=True)

# Function to plot sentiment distribution
def plot_sentiment_distribution(news):
    df = pd.DataFrame(news, columns=['Headline', 'Sentiment'])
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    # Map sentiment to colors
    sentiment_colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
    sentiment_counts['Color'] = sentiment_counts['Sentiment'].map(sentiment_colors)

    fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Color', color_discrete_map=sentiment_colors,
                 labels={'Count': 'Number of Headlines'},
                 title='Sentiment Distribution of Recent News Headlines 📈')

    # Customize x-axis labels
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['Postive', 'Negative', 'Neutral']
        ),
        showlegend=False
    )

    st.plotly_chart(fig)

# Function to get recent news headlines for a stock
def get_stock_news(ticker):
    
    api_key = '373064222bfa4a0abb23b5cab82c9cb5'  #** Please update the API key with your API for this tab to ensure its functioning"
    base_url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}'

    response = requests.get(base_url)
    data = response.json()

    # Extract news headlines and sentiment color
    headlines = [(article['title'], get_sentiment_color(analyze_sentiment(article['title']))) for article in data.get('articles', []) if ticker.lower() in article['title'].lower()]
    return headlines

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Classify the polarity of the sentiment
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to get color based on sentiment
def get_sentiment_color(sentiment):
    if sentiment == 'Positive':
        return 'green'
    elif sentiment == 'Negative':
        return 'red'
    else:
        return 'gray'

#***********************************************************************************************************************


def main():
    st.title("Stock Analytics Dashboard 🪩")
    col1, col2 = st.columns([1,5])
    col1.write("Data source:")
    col2.image('./img/yahoo_finance.png', width=100)
    

    selected_tab = st.selectbox("Select Tab:", ["Summary 📓", "Chart 🔮", "Financials 💷", "Monte Carlo Simulation ✔️","Stock Sentiment ☺️"])

    if selected_tab == "Summary 📓":
        # Call the function for the summary tab
        tab_summary()
    elif selected_tab == "Chart 🔮":
        # Call the function for the chart tab
        tab_chart()
    elif selected_tab == "Financials 💷":
        # Call the function for the financials tab
        tab_financials()
    elif selected_tab == "Monte Carlo Simulation ✔️":
        # Call the function for the Monte Carlo Simulation tab
        tab_mc_simulation()
    elif selected_tab == "Stock Sentiment ☺️":
        # Call the function for the stock sentiment analysis tab
        tab_stock_sentiment()


if __name__ == "__main__":
    main()