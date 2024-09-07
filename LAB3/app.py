import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
from scipy import stats

# Preprocessing Function
@st.cache_data
def preprocess_data():
    # Load the dataset
    data = pd.read_csv('covid_data.csv')
    
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # 'coerce' will convert invalid dates to NaT
    
    # Convert numeric columns to appropriate types, handling errors
    numeric_columns = ['Total Confirmed cases', 'Death', 'Cured/Discharged/Migrated', 'New cases', 'New deaths', 'New recovered']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)  # Fill NaNs with 0 and convert to int
    
    # Convert Latitude and Longitude to float if they exist
    if 'Latitude' in data.columns and 'Longitude' in data.columns:
        data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce').fillna(0)  # Handling any invalid values
        data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce').fillna(0)
    
    # Return the preprocessed data
    return data

# Load and preprocess data
data = preprocess_data()

# Title of the app
st.title("Comprehensive COVID-19 Data Dashboard: India")

# Key Metrics
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", data['Total Confirmed cases'].sum())
col2.metric("Total Deaths", data['Death'].sum())
col3.metric("Total Recovered", data['Cured/Discharged/Migrated'].sum())
col4.metric("New Cases (Most Recent)", data['New cases'].iloc[-1])

# Line Chart: Total Confirmed Cases Over Time
st.subheader("Line Chart: Total Confirmed Cases Over Time")
total_cases_time = data.groupby('Date')['Total Confirmed cases'].sum().reset_index()
fig_line = px.line(total_cases_time, x="Date", y="Total Confirmed cases",
                   title="Total Confirmed Cases Over Time")
st.plotly_chart(fig_line)

# Bar Chart: Total Cases by State/UT
st.subheader("Bar Chart: Total Confirmed Cases by State/UT")
total_cases_state = data.groupby('Name of State / UT')['Total Confirmed cases'].max().sort_values(ascending=False)
fig_bar = px.bar(x=total_cases_state.index, y=total_cases_state.values, 
                 labels={"x": "State/UT", "y": "Total Confirmed Cases"},
                 title="Total Confirmed Cases by State/UT")
st.plotly_chart(fig_bar)

# Pie Chart: Distribution of Cases by State/UT
st.subheader("Pie Chart: Distribution of Confirmed Cases by State/UT")
cases_distribution = data.groupby('Name of State / UT')['Total Confirmed cases'].sum()
fig_pie = px.pie(values=cases_distribution.values, names=cases_distribution.index, 
                 title="Distribution of Total Confirmed Cases by State/UT")
st.plotly_chart(fig_pie)

# Scatter Plot: Total Confirmed Cases vs Deaths
st.subheader("Scatter Plot: Total Confirmed Cases vs Deaths by State/UT")
fig_scatter = px.scatter(data, x='Total Confirmed cases', y='Death', 
                         color='Name of State / UT', size='Total Confirmed cases',
                         title="Total Confirmed Cases vs Deaths by State/UT",
                         labels={"Total Confirmed cases": "Total Confirmed Cases", "Death": "Total Deaths"})
st.plotly_chart(fig_scatter)

# Timeline Chart: New Cases Over Time
st.subheader("Timeline Chart: New Cases Over Time")
fig_timeline = px.scatter(data, x="Date", y="New cases", color="Name of State / UT",
                          title="New Cases Timeline by State/UT")
st.plotly_chart(fig_timeline)

# Histogram: Distribution of New Cases
st.subheader("Histogram: Distribution of New Cases")
fig_hist = px.histogram(data, x="New cases", nbins=20, title="Histogram of New Cases")
st.plotly_chart(fig_hist)

# Bubble Plot: New Cases vs Deaths vs Recoveries
st.subheader("Bubble Plot: New Cases vs Deaths vs Recoveries by State")
fig_bubble = px.scatter(data, x="New cases", y="Death", size="Cured/Discharged/Migrated", color="Name of State / UT",
                        hover_name="Name of State / UT", title="Bubble Plot: New Cases vs Deaths vs Recoveries")
st.plotly_chart(fig_bubble)

# Box Plot: Distribution of New Cases by State
st.subheader("Box Plot: Distribution of New Cases by State")
fig_box = px.box(data, x="Name of State / UT", y="New cases", 
                 title="Box Plot: Distribution of New Cases by State")
st.plotly_chart(fig_box)

# Violin Plot: Distribution of Deaths by State
st.subheader("Violin Plot: Distribution of Deaths by State")
fig_violin = px.violin(data, x="Name of State / UT", y="Death", box=True, points="all",
                       title="Violin Plot: Distribution of Deaths by State")
st.plotly_chart(fig_violin)

# Word Cloud: Generate a word cloud from the state names
st.subheader("Word Cloud: State Names")
state_text = ' '.join(data['Name of State / UT'])
wordcloud = WordCloud(background_color='white').generate(state_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Regression Plot: Linear (Total Cases vs Deaths)
st.subheader("Regression Plot: Linear Relationship between Total Cases and Deaths")
fig_reg = px.scatter(data, x="Total Confirmed cases", y="Death", trendline="ols", 
                     title="Linear Regression: Total Confirmed Cases vs Deaths")
st.plotly_chart(fig_reg)

# Regression Plot: Non-Linear (Total Cases vs Deaths with Non-Linear Fit)
st.subheader("Regression Plot: Non-Linear Relationship between Total Cases and Deaths")
plt.figure(figsize=(10, 6))
sns.regplot(x='Total Confirmed cases', y='Death', data=data, scatter_kws={'s': 20}, order=2)
plt.title("Non-Linear Regression: Total Confirmed Cases vs Deaths")
st.pyplot(plt.gcf())

# 3D Chart: Total Cases, Deaths, Recoveries by State
st.subheader("3D Chart: Total Cases, Deaths, Recoveries by State")
fig_3d = px.scatter_3d(data, x="Total Confirmed cases", y="Death", z="Cured/Discharged/Migrated", color="Name of State / UT",
                       title="3D Chart: Total Cases, Deaths, Recoveries by State")
st.plotly_chart(fig_3d)

# Jitter Plot using Seaborn (with Jitter)
st.subheader("Jitter Plot: New Cases by State")
plt.figure(figsize=(10, 6))
sns.stripplot(data=data, x="Name of State / UT", y="New cases", jitter=True)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title("Jitter Plot: New Cases by State")
st.pyplot(plt.gcf())

# Area Chart: Cumulative Recovered Cases Over Time
st.subheader("Area Chart: Cumulative Recovered Cases Over Time")
recovered_over_time = data.groupby('Date')['Cured/Discharged/Migrated'].sum().cumsum().reset_index(name='Cumulative Recovered')
fig_area = px.area(recovered_over_time, x="Date", y="Cumulative Recovered",
                   title="Cumulative Recovered Cases Over Time")
st.plotly_chart(fig_area)

# Waterfall Chart: New Cases Breakdown by State
st.subheader("Waterfall Chart: New Cases Breakdown by State")
waterfall_data = data.groupby('Name of State / UT')['New cases'].sum().sort_values(ascending=False)
fig_waterfall = go.Figure(go.Waterfall(
    name="New Cases", orientation="v",
    measure=["relative"] * len(waterfall_data),
    x=waterfall_data.index,
    y=waterfall_data.values,
    connector={"line": {"color": "rgb(63, 63, 63)"}}
))
fig_waterfall.update_layout(title="Waterfall Chart: New Cases Breakdown by State")
st.plotly_chart(fig_waterfall)

# Donut Chart: Deaths by State
st.subheader("Donut Chart: Deaths by State")
death_distribution = data.groupby('Name of State / UT')['Death'].sum()
fig_donut = px.pie(values=death_distribution.values, names=death_distribution.index, hole=0.4,
                   title="Donut Chart: Distribution of Deaths by State")
st.plotly_chart(fig_donut)

# Treemap: COVID-19 Cases Breakdown by State
st.subheader("Treemap: COVID-19 Cases Breakdown by State")
fig_treemap = px.treemap(data, path=['Name of State / UT'], values='Total Confirmed cases',
                         title="Treemap: COVID-19 Cases Breakdown by State")
st.plotly_chart(fig_treemap)

# Funnel Chart: COVID-19 Patient Flow
st.subheader("Funnel Chart: COVID-19 Patient Flow")
stages = ['Total Confirmed', 'Deaths', 'Recovered', 'Active Cases']
values = [
    data['Total Confirmed cases'].sum(),
    data['Death'].sum(),
    data['Cured/Discharged/Migrated'].sum(),
    data['Total Confirmed cases'].sum() - data['Death'].sum() - data['Cured/Discharged/Migrated'].sum()
]
fig_funnel = go.Figure(go.Funnel(
    y=stages, x=values,
    textinfo="value+percent initial",
))
fig_funnel.update_layout(title="Funnel Chart: COVID-19 Patient Flow")
st.plotly_chart(fig_funnel)

