import pandas as pd
import streamlit as st
from config import conf
from pathlib import Path
from prophet import Prophet
import plotly.express as px
from textblob import TextBlob
from collections import Counter

# Load the Excel file
df = pd.read_excel(Path(conf.DATA_DIR, "IT_Support_Chatbot_Dataset_200_Enhanced.xlsx"), engine = "openpyxl")

# Perform sentiment analysis
def get_sentiment_category(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df['sentiment_category'] = df['Feedback Received from User'].apply(get_sentiment_category)

# Count sentiment categories
sentiment_counts = df['sentiment_category'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'count']

# Create pie chart
fig_sentiment_pie = px.pie(sentiment_counts, names='sentiment', values='count', title='Sentiment Analysis of Feedback')

df['creation_time'] = pd.to_datetime(df['Open Time'])
df['closed_time'] = pd.to_datetime(df['Closure Time'])

# Forecasting
issue_counts = df.groupby(df['closed_time'].dt.date).size().reset_index(name='count')
issue_counts.columns = ['ds', 'y']
model = Prophet()
model.fit(issue_counts)
future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)
fig_forecast = px.line(forecast, x='ds', y='yhat')
fig_forecast.update_layout(xaxis_title='Date', yaxis_title='Predicted Issue Count')

# Sentiment analysis
def get_sentiment_category(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df['sentiment_category'] = df['Feedback Received from User'].apply(get_sentiment_category)
sentiment_counts = df['sentiment_category'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'count']
fig_sentiment_pie = px.pie(sentiment_counts, names='sentiment', values='count')

# Average resolution time
df['resolution_time_hours'] = (df['closed_time'] - df['creation_time']).dt.total_seconds() / 3600
avg_resolution_time = df['resolution_time_hours'].mean()

# Finding top 3 frequently occurring issues
if 'Issue Raised' in df.columns:
    # Drop NaNs and convert to list
    issues = df['Issue Raised'].dropna().tolist()

    # Count frequency
    issue_counts = Counter(issues)

    # Get top 3 issues
    top_3 = issue_counts.most_common(3)

    # Prepare data for Plotly
    issues_names = [issue for issue, count in top_3]
    issues_counts = [count for issue, count in top_3]
    data = pd.DataFrame({"Issue": issues_names, "Count": issues_counts})

    # Create Plotly bar chart
    fig_top3 = px.bar(data, x="Issue", y="Count", title="Top 3 Issues Raised", color="Issue",
                 labels={"Count": "Frequency", "Issue": "Issue"})

    # Display chart in Streamlit
    #st.plotly_chart(fig)

#print(df)
# Assignee distribution
assignee_counts =df['Assignee Name'].value_counts().reset_index()
assignee_counts.columns = ['Assignee Name', 'count']
fig_assignee = px.bar(assignee_counts, x='Assignee Name', y='count')
fig_assignee.update_layout(xaxis_title='Assignee Name', yaxis_title='Issues Count')

# Average resolution time per assignee
avg_time_per_assignee = df.groupby('Assignee Name')['resolution_time_hours'].mean().reset_index()
fig_avg_time_assignee = px.bar(
    avg_time_per_assignee,
    x='Assignee Name',
    y='resolution_time_hours'
)
fig_avg_time_assignee.update_layout(xaxis_title='Assignee Name', yaxis_title='Average Resolution Time (hours)')
print(df)
# Status distribution
status_counts = df['Resolution_status'].value_counts().reset_index()
status_counts.columns = ['status', 'count']
fig_status = px.pie(status_counts, names='status', values='count')

# Streamlit layout
st.title("OpsMate Analytics Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Forecasting",
    "Sentiment Analysis",
    "Assignee Overview",
    "Frequently Occurring Issues",
    "Resolution Time"
])

with tab1:
    st.subheader("Forecasted Issue Volume")
    st.plotly_chart(fig_forecast)

with tab2:
    st.subheader("Sentiment Analysis of Feedback")
    st.plotly_chart(fig_sentiment_pie)

with tab3:
    st.subheader("Issues being handled by each Assignee")
    st.plotly_chart(fig_assignee)
    st.subheader("Average Resolution Time per Assignee")
    st.plotly_chart(fig_avg_time_assignee)

with tab4:
    st.subheader("Frequently Occurring Issues")
    st.plotly_chart(fig_top3)

with tab5:
    st.subheader("Average Resolution Time")
    st.write(f"Average resolution time: {avg_resolution_time:.2f} hours")


############################################ Subham ####################################################


# import streamlit as st
# import pandas as pd
# import pyttsx3
# import matplotlib.pyplot as plt

# # --- Voice Assistant Function ---
# def speak(text):
#     engine = pyttsx3.init()
#     voices = engine.getProperty('voices')
#     # Set to female voice if available
#     for voice in voices:
#         if "female" in voice.name.lower() or "zira" in voice.name.lower():
#             engine.setProperty('voice', voice.id)
#             break
#     engine.say(text)
#     engine.runAndWait()

# # --- Load Data ---
# df = pd.read_csv("<path>/defect_data.csv")

# st.title("Innovative Defect Analytics Dashboard with Voice Assistant")

# st.sidebar.header("Filter Data")
# selected_team = st.sidebar.multiselect("Team", options=df["Team"].unique(), default=list(df["Team"].unique()))
# selected_severity = st.sidebar.multiselect("Severity", options=df["Severity"].unique(), default=list(df["Severity"].unique()))
# selected_priority = st.sidebar.multiselect("Priority", options=df["Priority"].unique(), default=list(df["Priority"].unique()))
# selected_component = st.sidebar.multiselect("Component", options=df["Component"].unique(), default=list(df["Component"].unique()))

# filtered_df = df[
#     df["Team"].isin(selected_team) &
#     df["Severity"].isin(selected_severity) &
#     df["Priority"].isin(selected_priority) &
#     df["Component"].isin(selected_component)
# ]

# st.subheader("Defect Data Preview")
# st.dataframe(filtered_df)

# st.subheader("Defect Count by Component")
# st.bar_chart(filtered_df["Component"].value_counts())

# st.subheader("Defect Severity Distribution")
# st.bar_chart(filtered_df["Severity"].value_counts())

# st.subheader("Defect Priority Distribution")
# st.bar_chart(filtered_df["Priority"].value_counts())

# st.subheader("Defect Trend Over Time")
# trend_df = filtered_df.groupby("Creation time").size().reset_index(name="Defect Count")
# trend_df["Creation time"] = pd.to_datetime(trend_df["Creation time"])
# trend_df = trend_df.sort_values("Creation time")
# st.line_chart(trend_df.set_index("Creation time"))

# st.subheader("Top 5 Recent Critical Defects")
# critical_defects = filtered_df[filtered_df["Severity"] == "Critical"].sort_values("Creation time", ascending=False).head(5)
# st.table(critical_defects[["ID", "Name", "Component", "Priority", "Owner", "Creation time"]])

# st.subheader("Defect Root Cause Analysis")
# st.bar_chart(filtered_df["Defect Root Cause"].value_counts())

# st.subheader("Defect Resolution Time Analysis")
# filtered_df["Resolution Time (days)"] = (
#     pd.to_datetime(filtered_df["Estimated Fix Date"]) - pd.to_datetime(filtered_df["Creation time"])
# ).dt.days

# if not filtered_df["Resolution Time (days)"].dropna().empty:
#     fig, ax = plt.subplots()
#     ax.boxplot(filtered_df["Resolution Time (days)"].dropna())
#     ax.set_title("Resolution Time (days) Box Plot")
#     st.pyplot(fig)
# else:
#     st.info("No data available for box plot.")

# st.markdown("---")

# # --- Voice Assistant Section ---
# summary = f"There are {len(filtered_df)} defects in the current view. " \
#           f"Top component is {filtered_df['Component'].value_counts().idxmax()} with {filtered_df['Component'].value_counts().max()} defects." \
#           f" Most common root cause is {filtered_df['Defect Root Cause'].value_counts().idxmax()}."

# st.subheader("Voice Assistant")
# st.write(summary)

# if st.button("🔊 Read Summary"):
#     speak(summary)

# st.info("Use the sidebar to filter defects. Click 'Read Summary' to hear the dashboard insights.")

# # Instructions:
# # pip install streamlit pandas pyttsx3 matplotlib
# # Run with: streamlit run