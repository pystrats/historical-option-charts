import streamlit as st
import pandas as pd
import math
from pathlib import Path

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

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :chart_with_upwards_trend: Historical SPXW Charts
'''
# Add some spacing

timeframes = {f"{min} minute(s)":min  for min in [1, 2, 3, 5, 10, 15, 30, 60]}
error_msg = ""
error = st.error(error_msg, icon="🚨")
error.empty()

chart = StreamlitChart(height=800)
chart.layout(background_color='#0e1118')

df = pd.read_csv(Path(__file__).parent/'data/ohlc.csv')
chart.set(df)

def go(): 
    error.empty()
    error_msg = st.session_state.TIMEFRAME
    st.error(error_msg, icon="🚨")

col1, col2= st.columns(2)
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

''


chart.load()




min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )


