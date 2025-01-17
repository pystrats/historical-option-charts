import streamlit as st
import pandas as pd
import math
from pathlib import Path
from datetime import datetime, timedelta

import polygon
from polygon.options.options import OptionsClient

from pytz import timezone
eastern = timezone('US/Eastern')

import pandas as pd
from lightweight_charts.widgets import StreamlitChart

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Historical SPXW Charts',
    page_icon=':chart_with_upwards_trend:', 
    layout='wide',
)

'''
# :chart_with_upwards_trend: Historical SPXW Charts
'''
def custom_error(error_msg:str):
    st.error(error_msg, icon="🚨")

timeframes = {f"{min} minute(s)":min  for min in [1, 2, 3, 5, 10, 15, 30, 60]}
error_msg = ""
error = st.error(error_msg)
error.empty()



df = pd.read_csv(Path(__file__).parent/'data/ohlc.csv')

def RMI(bars):

    RMI_LENGTH = 9
    RMI_MOMENTUM_LENGTH = 5

    closes = list(bars['close'])
    highs = list(bars['high'])
    lows = list(bars['low'])
    _RMI = [None for _ in range(len(closes))]

    def RMI_value(up, down): 
        try:
            return (100. if down == 0. else 0. if up == .0 else 100. - (100. / (1. + up / down)))
        except: return None

    n = 0
    first_valid_index = RMI_LENGTH + RMI_MOMENTUM_LENGTH
    up_raw, down_raw = [], []
    up, down = 0., 0.
    for i in range(len(closes)):
        if n >= RMI_MOMENTUM_LENGTH: 
            up_raw.append(max(closes[n] - closes[n-RMI_MOMENTUM_LENGTH], 0))
            down_raw.append(-1.*min(closes[n] - closes[n-RMI_MOMENTUM_LENGTH], 0))
        if n == first_valid_index:
            up, down = sum(up_raw)/len(up_raw), sum(down_raw)/len(down_raw)
            _RMI[n] = RMI_value(up, down)
        elif n > first_valid_index:
            up = (up * (RMI_LENGTH - 1) + up_raw[-1]) / RMI_LENGTH
            down = (down * (RMI_LENGTH - 1) + down_raw[-1]) / RMI_LENGTH
            _RMI[n] = RMI_value(up, down)
        n += 1


    _df = pd.DataFrame({
        'time': bars['time'],
        f'RMI': _RMI
    })
    return _df

def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_percent_rank(series, window):
    return series.rolling(window).apply(lambda x: (x.rank(pct=True).iloc[-1]) * 100, raw=False)

def connors_rsi(data, rsi_period=3, streak_rsi_period=2, percent_rank_period=100):
    # Calculate the RSI
    rsi = compute_rsi(data['close'], rsi_period)
    
    # Calculate the streak
    streak = (data['close'] - data['close'].shift(1)).apply(
        lambda x: streak + 1 if x > 0 else streak - 1 if x < 0 else 0
    )
    
    # Calculate the Streak RSI
    streak_rsi = compute_rsi(streak, streak_rsi_period)
    
    # Calculate the Percent Rank of the 1-day change
    percent_rank = compute_percent_rank(data['close'].diff(), percent_rank_period)
    
    # Combine all components
    connors_rsi = (rsi + streak_rsi + percent_rank) / 3

    df = pd.DataFrame({
        'time': data['time'],
        f'ConnorsRSI': connors_rsi
    })
    return connors_rsi


def go(): 
    error.empty()

    if st.session_state.API_KEY == '':
        custom_error("Please provide API key")
        return

    try:
        strike = float(st.session_state.STRIKE)
    except:
        custom_error("Invalid strike")
        return

    client = OptionsClient(api_key=API_KEY)

    last_date = st.session_state.EXPIRATION_DATE
    symbol = polygon.build_polygon_option_symbol(
        'SPXW',
        last_date,
        call_or_put=st.session_state.RIGHT.lower(),
        strike_price=strike
    )

    tf = timeframes.get(st.session_state.TIMEFRAME, 1)
    bars = client.get_aggregate_bars(
                symbol=symbol,
                from_date=last_date-timedelta(days=5 if tf in [1, 2, 3] else 50 if tf in [5, 10, 15] else 125),
                multiplier=tf,
                to_date=last_date,
                limit=10000,
                timespan='minute',
                full_range=True,
                run_parallel=True,
                max_concurrent_workers=10,
                high_volatility=True
            )

    
    if bars == []:
        custom_error("Data request failed. Please provide a valid API key and contract specifications.")
        return
    
    _dict=[]
    for bar in bars:
        dt = datetime.fromtimestamp(int(bar['t'])/1000).astimezone(eastern)
        _dict.append([
            dt.strftime("%Y-%m-%d %H:%M:%S"),
            bar['o'],
            bar['h'],
            bar['l'],
            bar['c'],
            bar['v']
        ])
    df = pd.DataFrame(_dict, columns=['time','open','high','low','close','volume'])
    with container:
        charts = set()
        chart = StreamlitChart(height=st.session_state.CHART_HEIGHT, toolbox=False, inner_width=1, inner_height=0.6)
        charts.add(chart)
        chart.time_scale(visible=False)
        chart.legend(visible=True, text=symbol, font_size=16, color_based_on_candle=True)
        chart.crosshair('magnet')

        chart2 = chart.create_subchart(width=1, height=0.2, sync=True, sync_crosshairs_only=False)
        chart2.legend(visible=True, text='RMI', font_size=16, lines=True)
        chart2.crosshair('magnet')
        charts.add(chart2)

        chart3 = chart.create_subchart(width=1, height=0.2, sync=True, sync_crosshairs_only=False)
        chart3.legend(visible=True, text='ConnorsRSI', font_size=16, lines=True)
        chart3.crosshair('magnet')
        charts.add(chart3)
        
        line = chart2.create_line(name='RMI')
        upper_threshold = chart2.create_line(name='Upper Threshold', price_label = False, color='#32a852')
        lower_threshold = chart2.create_line(name='Lower Threshold', price_label = False, color='#32a852')

        connors_line = chart3.create_line(name='ConnorsRSI')

        for _chart in charts: _chart.layout(background_color='#0e1118')

        chart.set(df)
        line.set(RMI(df))
        upper_threshold.set(pd.DataFrame({
            'time': df['time'],
            'Upper Threshold': [st.session_state.UP_THRESHOLD for _ in range(df.shape[0])]
        }))
        lower_threshold.set(pd.DataFrame({
            'time': df['time'],
            'Lower Threshold': [st.session_state.LO_THRESHOLD for _ in range(df.shape[0])]
        }))
        connors_line.set(connors_rsi(df))
        chart.load()
    
    

col1, col2, col3 = st.columns(3)
with col1:
    st.text_input("Polygon API key", key="API_KEY")
    API_KEY = st.session_state.API_KEY

    st.date_input('Expiration date', key='EXPIRATION_DATE')

    right = st.selectbox(
        "Right",
        ("CALL", "PUT"),
        key="RIGHT"
    )

with col2:
    st.text_input("Strike", key="STRIKE")
    
    st.selectbox(
        "Timeframe",
        (k for k, _ in timeframes.items()),
        key="TIMEFRAME"
    )

    
    st.number_input('Chart Height, px', min_value=250, max_value=3000, value=900, step=10, key='CHART_HEIGHT')
    
    

    

with col3:

    st.number_input('Upper Threshold', min_value=50, max_value=100, value=90, step=1, key='UP_THRESHOLD')

    st.number_input('Lower Threshold', min_value=0, max_value=50, value=20, step=1, key='LO_THRESHOLD')

    ''
    ''

    col3.button("Go", use_container_width=True, type="primary", on_click=go)

container = st.container()
''

