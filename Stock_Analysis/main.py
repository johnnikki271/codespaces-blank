import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from claude_api import Client
import google.generativeai as genai
import os
import csv
from json import JSONDecoder
from dotenv import load_dotenv

load_dotenv()
claude_api = Client(os.environ["CLAUDE_COOKIE"])

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)


def get_stock_data(symbol, start_date, end_date):
  """Fetch stock data from Yahoo Finance"""
  try:
    return yf.download(symbol, start=start_date, end=end_date)
  except Exception as e:
    print(f"Error downloading {symbol} data: {e}")
    return None


def calculate_indicators(df):
  """Calculate various technical indicators"""
  df.ta.macd(append=True)
  df.ta.rsi(append=True)
  df.ta.bbands(append=True)
  df.ta.obv(append=True)

  df.ta.sma(length=20, append=True)
  df.ta.ema(length=50, append=True)
  df.ta.stoch(append=True)
  df.ta.adx(append=True)

  df.ta.willr(append=True)
  df.ta.cmf(append=True)
  df.ta.psar(append=True)

  # Additional calculations
  df['OBV_in_million'] = df['OBV'] / 1e7
  df['MACD_histogram_12_26_9'] = df['MACDh_12_26_9']

  return df


def get_last_day_summary(df):
  """Return summary of indicators for the last day"""
  return df.iloc[-1][[
      'Adj Close', 'MACD_12_26_9', 'MACD_histogram_12_26_9', 'RSI_14',
      'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'SMA_20', 'EMA_50',
      'OBV_in_million', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ADX_14', 'WILLR_14',
      'CMF_20', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2'
  ]]


def generate_prompt(symbol, indicators):
  """Generate prompt for Claude based on indicators"""
  return """
  Assume the role as a leading Technical Analysis (TA) expert in the stock market,
  a modern counterpart to Charles Dow, John Bollinger, and Alan Andrews.
  Your mastery encompasses both stock fundamentals and intricate technical indicators.
  You possess the ability to decode complex market dynamics,
  providing clear insights and recommendations backed by a thorough understanding of interrelated factors.
  Your expertise extends to practical tools like the pandas_ta module,
  allowing you to navigate data intricacies with ease.
  As a TA authority, your role is to decipher market trends, make informed predictions, and offer valuable perspectives.

  given {} TA data as below on the last trading day, what will be the next few days possible stock price movement? 

  Summary of Technical Indicators for the Last Day:
  {}""".format(symbol, indicators)

def delete_conversation(conversation_id):
    deleted = claude_api.delete_conversation(conversation_id)
    if deleted:
        print("Conversation deleted successfully")
        return True
    else:
        print("Failed to delete conversation")
        return False
    

def create_conversation(claude_api):
  """Create new conversation and return ID"""
  return claude_api.create_new_chat()['uuid']


def send_message(claude_api, prompt, conversation_id):
  """Send message to existing conversation"""
  return claude_api.send_message(prompt, conversation_id)


def get_claude_prediction(prompt):
  """Get prediction from Claude"""
  # Create conversation
  global conversation_id
  conversation_id = create_conversation(claude_api)
  # Send message
  response = send_message(claude_api, prompt, conversation_id)

  return response


def parse_claude_prediction(symbol, prompt, response):
  """Parse Claude response into structured JSON"""
  parse_prompt = """Based on the below statement output me in
  JSON format as shown in example below
  Example Response 1 : {
  "Symbol" : "YESBANK.NS",
  "Result" : ["Bullish"]
  }
  Example Response 2 :{
  "Symbol" : "YESBANK.NS",
  "Result" : ["Bearish"]
  }
  Example Response 3 :{
  "Symbol" : "YESBANK.NS",
  "Result" : ["Sideways"]
  }
  Example Response 4 :{
  "Symbol" : "YESBANK.NS",
  "Result" : ["Sideways","Bearish"]
  }
  Example Response 5 :{
  "Symbol" : "YESBANK.NS",
  "Result" : ["Sideways","Bullish"]
  }

  System Instrucion : *** STRICTLY DONT OUTPUT ANY OTHER TEXT OTHER THAN JSON ***.

  Q:
  """ + prompt + """ 
  """ + "Claude Response: {}".format(response) + """

  Ans:"""
  parse_claude_verdict = send_message(claude_api, parse_prompt,
                                      conversation_id)

  # Some logic here to extract prediction
  return parse_claude_verdict


def get_gemini_prediction(prompt):

  model = genai.GenerativeModel('gemini-pro')

  # Send message
  response = model.generate_content(prompt)

  return response.text


def parse_gemini_prediction(symbol, prompt, response):
  """Parse Claude response into structured JSON"""
  model = genai.GenerativeModel('gemini-pro')
  gemini_parse_prompt = """Based on the below statement output me in
  JSON format as shown in example below
  Example Response 1 : {
  "Symbol" : "YESBANK.NS",
  "Result" : ["Bullish"]
  }
  Example Response 2 :{
  "Symbol" : "YESBANK.NS",
  "Result" : ["Bearish"]
  }
  Example Response 3 :{
  "Symbol" : "YESBANK.NS",
  "Result" : ["Sideways"]
  }
  Example Response 4 :{
  "Symbol" : "YESBANK.NS",
  "Result" : ["Sideways","Bearish"]
  }
  Example Response 5 :{
  "Symbol" : "YESBANK.NS",
  "Result" : ["Sideways","Bullish"]
  }

  System Instrucion : *** STRICTLY DONT OUTPUT ANY OTHER TEXT OTHER THAN JSON ***.

  Q:
  """ + prompt + """ 
  """ + "Gemini Response: {}".format(response) + """

  Ans:"""
  parse_gemini_verdict = model.generate_content(gemini_parse_prompt)
  # Some logic here to extract prediction
  return parse_gemini_verdict.text

def print_verdict(verdict_name, verdict):
    # Usage: Extract JSON from structured data
    json_output = None
    for result in extract_json_objects(verdict):
        json_output = result

    if json_output is not None:
        symbol = list(json_output.items())[0][1]
        result = list(json_output.items())[1][1]
        prediction = ""
        prediction = ",".join(str(r) for r in result)
    else:
        symbol = "N/A"
        prediction = "No JSON object found in the text."
    print(f"{verdict_name:<15} {symbol:<15} {prediction}")

def visualize(stock_data):
  # Plot the technical indicators
  plt.figure(figsize=(14, 8))

  # Price Trend Chart
  plt.subplot(3, 3, 1)
  plt.plot(stock_data.index,
           stock_data['Adj Close'],
           label='Adj Close',
           color='blue')
  plt.plot(stock_data.index,
           stock_data['EMA_50'],
           label='EMA 50',
           color='green')
  plt.plot(stock_data.index,
           stock_data['SMA_20'],
           label='SMA_20',
           color='orange')
  plt.title("Price Trend")
  plt.gca().xaxis.set_major_formatter(
      mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
  plt.xticks(rotation=45, fontsize=8)  # Adjust font size
  plt.legend()

  # On-Balance Volume Chart
  plt.subplot(3, 3, 2)
  plt.plot(stock_data['OBV'], label='On-Balance Volume')
  plt.title('On-Balance Volume (OBV) Indicator')
  plt.gca().xaxis.set_major_formatter(
      mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
  plt.xticks(rotation=45, fontsize=8)  # Adjust font size
  plt.legend()

  # MACD Plot
  plt.subplot(3, 3, 3)
  plt.plot(stock_data['MACD_12_26_9'], label='MACD')
  plt.plot(stock_data['MACDh_12_26_9'], label='MACD Histogram')
  plt.title('MACD Indicator')
  plt.gca().xaxis.set_major_formatter(
      mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
  plt.xticks(rotation=45, fontsize=8)  # Adjust font size
  plt.title("MACD")
  plt.legend()

  # RSI Plot
  plt.subplot(3, 3, 4)
  plt.plot(stock_data['RSI_14'], label='RSI')
  plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
  plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
  plt.legend()
  plt.gca().xaxis.set_major_formatter(
      mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
  plt.xticks(rotation=45, fontsize=8)  # Adjust font size
  plt.title('RSI Indicator')

  # Bollinger Bands Plot
  plt.subplot(3, 3, 5)
  plt.plot(stock_data.index, stock_data['BBU_5_2.0'], label='Upper BB')
  plt.plot(stock_data.index, stock_data['BBM_5_2.0'], label='Middle BB')
  plt.plot(stock_data.index, stock_data['BBL_5_2.0'], label='Lower BB')
  plt.plot(stock_data.index,
           stock_data['Adj Close'],
           label='Adj Close',
           color='brown')
  plt.title("Bollinger Bands")
  plt.gca().xaxis.set_major_formatter(
      mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
  plt.xticks(rotation=45, fontsize=8)  # Adjust font size
  plt.legend()

  # Stochastic Oscillator Plot
  plt.subplot(3, 3, 6)
  plt.plot(stock_data.index, stock_data['STOCHk_14_3_3'], label='Stoch %K')
  plt.plot(stock_data.index, stock_data['STOCHd_14_3_3'], label='Stoch %D')
  plt.title("Stochastic Oscillator")
  plt.gca().xaxis.set_major_formatter(
      mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
  plt.xticks(rotation=45, fontsize=8)  # Adjust font size
  plt.legend()

  # Williams %R Plot
  plt.subplot(3, 3, 7)
  plt.plot(stock_data.index, stock_data['WILLR_14'])
  plt.title("Williams %R")
  plt.gca().xaxis.set_major_formatter(
      mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
  plt.xticks(rotation=45, fontsize=8)  # Adjust font size

  # ADX Plot
  plt.subplot(3, 3, 8)
  plt.plot(stock_data.index, stock_data['ADX_14'])
  plt.title("Average Directional Index (ADX)")
  plt.gca().xaxis.set_major_formatter(
      mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
  plt.xticks(rotation=45, fontsize=8)  # Adjust font size

  # CMF Plot
  plt.subplot(3, 3, 9)
  plt.plot(stock_data.index, stock_data['CMF_20'])
  plt.title("Chaikin Money Flow (CMF)")
  plt.gca().xaxis.set_major_formatter(
      mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
  plt.xticks(rotation=45, fontsize=8)  # Adjust font size

  # Show the plots
  plt.tight_layout()
  plt.show()


def extract_json_objects(text, decoder=JSONDecoder()):
  pos = 0
  while True:
    match = text.find('{', pos)
    if match == -1:
      break
    try:
      result, index = decoder.raw_decode(text[match:])
      yield result
      pos = match + index
    except ValueError:
      pos = match + 1

def process_symbol(symbol, start_date, end_date):
  try:

    df = get_stock_data(symbol, start_date, end_date)
    df = calculate_indicators(df)

    indicators = get_last_day_summary(df)
    prompt = generate_prompt(symbol, indicators)
    print(prompt)
    print("")
    claude_prediction = get_claude_prediction(prompt)
    print("Claude Reason : ")
    print("")
    print(claude_prediction)
    print("")
    gemini_prediction = get_gemini_prediction(prompt)
    print("Gemini Reason : ")
    print("")
    print(gemini_prediction)
    print("")
    claude_verdict = parse_claude_prediction(symbol, prompt, claude_prediction)
    print(claude_verdict)
    print("")
    gemini_verdict = parse_gemini_prediction(symbol, prompt, gemini_prediction)
    print(gemini_verdict)
    print("")
    print('-' * 50)
    print(f"{'AI Model':<15} {'Symbol':<15} {'Prediction'}")
    print('-' * 50)
    print_verdict("Claude", claude_verdict)
    print_verdict("Gemini-Pro", gemini_verdict)
    print('-' * 50)
    print("")
  except Exception as e:
    print(f"Error processing {symbol}: {e}")
  finally:
    delete_conversation(conversation_id)


def main():
  try:
    end_date = datetime.today()
    #end_date = datetime(2020, 4, 30)
    start_date = end_date - timedelta(days=120)
    #symbol = "^NSEI"
    symbol = "YESBANK.NS"
    # Process NSE index
    process_symbol(symbol, start_date, end_date)
    # Process each stock
    # with open('stocks.csv', 'r') as f:
    #   reader = csv.DictReader(f)
    #   for row in reader:
    #     symbol = row['Symbol'] + ".NS"
    #     process_symbol(symbol, start_date, end_date)
  except Exception as e:
    print(f"Error processing {e}")


if __name__ == "__main__":
  main()
