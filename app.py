#!/usr/bin/env python

from pymongo import MongoClient
from helper_functions import _get_config
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Event

parser = _get_config()
topic = parser.get('TOPIC', 'topic')
mongo_host = parser.get('Mongodb', 'host')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df_new = pd.DataFrame(columns=["sentiment_mean", "time"])

app.layout = html.Div([
    html.H1("Real-time Twitter Sentiment Prediction"),
    html.H2("Topic: " + topic),
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(id='graph-update', interval=1*1000), #update every 1 seconds
])

@app.callback(Output('live-graph', 'figure'),events=[Event('graph-update', 'interval')])
def update_graph_scatter():
    try:
        client = MongoClient(mongo_host)
        db = client.twitter_sentiment
        collection_name = "twitter_sentiment_" + topic
        db_coll = db[collection_name]
        tweets_recent_10000 = db_coll.find().sort("$natural", -1).limit(10000)
        df = pd.DataFrame(list(tweets_recent_10000))
        #df["tweeted_at"] = df.tweeted_at.astype("datetime64[ns]")
        df["insert_time"] = df.insert_time.astype("float64")
        df["sentiment_10000"] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df.dropna(inplace=True)

        X = df.insert_time[-1000:]
        Y = df.sentiment_10000[-1000:]

        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                mode= 'lines+markers'
                )

        return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)], showticklabels=False),
                                                    yaxis=dict(range=[min(Y),max(Y)]),)}

    except Exception as e:
        print e

if __name__ == '__main__':
    app.run_server(debug=True)
