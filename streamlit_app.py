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
timeframes = {f"{min} minute(s)":min  for min in [1, 2, 3, 5, 10, 15, 30, 60]}
error_msg = ""
error = st.error(error_msg, icon="🚨")
error.empty()



df = pd.read_csv(Path(__file__).parent/'data/ohlc.csv')

def go(): 
    error.empty()

    if st.session_state.API_KEY == '':
        st.error("Please provide API key", icon="🚨")
        return

    try:
        strike = float(st.session_state.STRIKE)
    except:
        st.error("Invalid strike", icon="🚨")
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
        st.error("No data found. Please provide a valid API key and contract specifications.", icon="🚨")
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
        chart = StreamlitChart(height=800, toolbox=True)
        chart.layout(background_color='#0e1118')
        chart.legend(visible=True, text=symbol, font_size=16, color_based_on_candle=True)
        chart.set(df)  
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
    ''
    ''
    

    col2.button("Go", use_container_width=True, type="primary", on_click=go)

container = st.container()
''

