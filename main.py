
import dash
from dash import dcc as dcc
from dash import html as html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import RobustScaler
from plotly.tools import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from google.cloud import storage

import seaborn as sns
import matplotlib.pyplot as plt

client = storage.Client()
bucket = client.get_bucket('andromeda-data')

ccpiBlob = bucket.blob('ccpi_m.csv')
ecpiBlob = bucket.blob('ecpi_m.csv')
fcpiBlob = bucket.blob('fcpi_m.csv')
hcpiBlob = bucket.blob('hcpi_m.csv')
ppiBlob = bucket.blob('ppi_m.csv')

ccpiData = pd.read_csv(ccpiBlob.download_as_string())
ecpiData = pd.read_csv(ecpiBlob.download_as_string())
fcpiData = pd.read_csv(fcpiBlob.download_as_string())
hcpiData = pd.read_csv(hcpiBlob.download_as_string())
ppiData = pd.read_csv(ppiBlob.download_as_string())


ccpiCountryCodes = ccpiData[['IMF Country Code','Country Code','Country']]
ecpiCountryCodes = ecpiData[['IMF Country Code','Country Code','Country']]
fcpiCountryCodes = fcpiData[['IMF Country Code','Country Code','Country']]
hcpiCountryCodes = hcpiData[['IMF Country Code','Country Code','Country']]
ppiCountryCodes = ppiData[['IMF Country Code','Country Code','Country']]


ccpiCountryCodes.columns = [col + ' CCPI' for col in ccpiCountryCodes.columns]
ecpiCountryCodes.columns = [col + ' ECPI' for col in ecpiCountryCodes.columns]
fcpiCountryCodes.columns = [col + ' FCPI' for col in fcpiCountryCodes.columns]
hcpiCountryCodes.columns = [col + ' HCPI' for col in hcpiCountryCodes.columns]
ppiCountryCodes.columns = [col + ' PPI' for col in ppiCountryCodes.columns]


commonCountryCodes = ccpiCountryCodes.join(ecpiCountryCodes.set_index('IMF Country Code ECPI'), on='IMF Country Code CCPI', how = 'inner').join(fcpiCountryCodes.set_index('IMF Country Code FCPI'), on='IMF Country Code CCPI', how = 'inner').join(hcpiCountryCodes.set_index('IMF Country Code HCPI'), on='IMF Country Code CCPI', how = 'inner').join(ppiCountryCodes.set_index('IMF Country Code PPI'), on='IMF Country Code CCPI', how = 'inner').drop_duplicates(subset='IMF Country Code CCPI')


commonCountryCodesList = commonCountryCodes['Country Code CCPI'].tolist()
commonCountryNamesList = commonCountryCodes['Country CCPI'].tolist()


def segregate_train_test_data(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)


client = storage.Client()
bucket = client.get_bucket('andromeda-data')

resultBlob = bucket.blob('lstmHyperParameters.csv')

result = pd.read_csv(resultBlob.download_as_string())
result.set_index('index', inplace=True)


result_dict = result.to_dict(orient='index')


if __name__ == '__main__':

    app = dash.Dash()

    def readCountryData(country_code):

        ccpiBlob = bucket.blob('ccpi_m.csv')
        ecpiBlob = bucket.blob('ecpi_m.csv')
        fcpiBlob = bucket.blob('fcpi_m.csv')
        hcpiBlob = bucket.blob('hcpi_m.csv')
        ppiBlob = bucket.blob('ppi_m.csv')

        ccpiData = pd.read_csv(ccpiBlob.download_as_string())
        ecpiData = pd.read_csv(ecpiBlob.download_as_string())
        fcpiData = pd.read_csv(fcpiBlob.download_as_string())
        hcpiData = pd.read_csv(hcpiBlob.download_as_string())
        ppiData = pd.read_csv(ppiBlob.download_as_string())

        rawDF = pd.concat([ccpiData, ecpiData, fcpiData, hcpiData, ppiData])
        rawDF = rawDF.T
        rawDF.rename(columns=rawDF.iloc[0,:], inplace = True)
        rawDF = rawDF.tail(-1)
        rawDF.drop(rawDF.tail(1).index,inplace=True)
        time = pd.DatetimeIndex([i[:-2]+'-'+i[-2:] for i in rawDF.index])
        rawDF = rawDF.set_index(time)
        df = rawDF.fillna(method='ffill').fillna(method='bfill')

        dataGreaterthan2003 = df[df.index >= '2003-01-01']
        return dataGreaterthan2003

    structureDF = readCountryData('ALB')
    
    countryCodeDropdown = html.Div([dcc.Dropdown(
        id='country-selection-dropdown',
        options=[{'label': i, 'value': j} for i, j in zip(commonCountryNamesList, commonCountryCodesList)],
        value=commonCountryCodesList[0]
        )])
    displayCountryName = html.Div(id='country-selected')
    displaySelectedMonth = html.Div(id='world-map-month-selected')
    displayPredictionMonths = html.Div(id='prediction-months-selected')
    precisionDropdown =  html.Div([
        dcc.Dropdown(
            id='precision-selection-dropdown',
            options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'High', 'value': 'high'}
            ],
            value='low',
            style={
                'width': '25%',
                'margin': 'auto'
            }
        )
    ], style={
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center'
    })

    @app.callback(dash.dependencies.Output('country-selected', 'children'),
                dash.dependencies.Input('country-selection-dropdown', 'value'))
    def update_output(value):
        country_code = value
        return html.H3('Country Selected is : ' + commonCountryNamesList[commonCountryCodesList.index(country_code)],
            style={'textAlign': 'center',
                    'color': '#000205'}
            )

    @app.callback(dash.dependencies.Output('line-plot', 'figure'),
                [dash.dependencies.Input('year-range-slider-line-plot', 'value'), 
                 dash.dependencies.Input('country-selection-dropdown', 'value')])
    def update_line_plot(year_range, value):
        country_code = value
        df = readCountryData(country_code)

        df_robust = df.copy()
        # apply robust scaling
        for column in df_robust.columns:
            df_robust[column] = (df_robust[column] - df_robust[column].median())  / (df_robust[column].quantile(0.75) - df_robust[column].quantile(0.25))

        df = df_robust
        
        # filter the DataFrame by the selected year range
        filtered_df = df[(df.index.year >= year_range[0]) & (df.index.year <= year_range[1])]
        
        df = filtered_df
        
        # Create a line plot of the dataframe using go.Scatter for each column
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))

        # Set the layout for the plot
        fig.update_layout(title='Line Plot of inflation indices',
                        xaxis_title='Year', yaxis_title='Value')
        
        return fig    

    @app.callback(dash.dependencies.Output('histogram', 'figure'),
                [dash.dependencies.Input('year-range-slider-histogram', 'value'), 
                 dash.dependencies.Input('country-selection-dropdown', 'value')])
    def update_histogram(year_range, value):
        country_code = value
        df = readCountryData(country_code)

        df_robust = df.copy()
        # apply robust scaling
        for column in df_robust.columns:
            df_robust[column] = (df_robust[column] - df_robust[column].median())  / (df_robust[column].quantile(0.75) - df_robust[column].quantile(0.25))

        df = df_robust
        
        # filter the DataFrame by the selected year range
        filtered_df = df[(df.index.year >= year_range[0]) & (df.index.year <= year_range[1])]
        
        traces = []
        for col in filtered_df.columns:
            trace = go.Histogram(x=filtered_df[col], name=col, opacity=0.3, nbinsx=30)
            traces.append(trace)
        
        layout = go.Layout(title='Histogram plot of inflation indices',barmode='overlay')
        
        fig = go.Figure(data=traces, layout=layout)
        
        return fig
   
    @app.callback(dash.dependencies.Output('inflation-violin-plot', 'figure'),
                [dash.dependencies.Input('year-range-slider-violin', 'value'), 
                 dash.dependencies.Input('country-selection-dropdown', 'value')])
    def update_violin_plot(year_range, value):
        
        country_code = value
        df = readCountryData(country_code)

        df_robust = df.copy()
        # apply robust scaling
        for column in df_robust.columns:
            df_robust[column] = (df_robust[column] - df_robust[column].median())  / (df_robust[column].quantile(0.75) - df_robust[column].quantile(0.25))

        df = df_robust
    
        # filter the DataFrame by the selected year range
        filtered_df = df[(df.index.year >= year_range[0]) & (df.index.year <= year_range[1])]
        # create a list of violin traces for each column
        traces = [go.Violin(y=filtered_df[col], name=col) for col in filtered_df.columns]
        # create the figure object and return it
        fig = go.Figure(data=traces)
        fig.update_layout(title='Inflation Indices Violin Plots for Selected Year Range')
        return fig

    @app.callback(dash.dependencies.Output('pair-plot', 'figure'),
                [dash.dependencies.Input('year-range-slider-pair-plot', 'value'), 
                 dash.dependencies.Input('country-selection-dropdown', 'value')])
    def update_pair_plot(year_range, value):
        country_code = value
        df = readCountryData(country_code)

        df_robust = df.copy()
        # apply robust scaling
        for column in df_robust.columns:
            df_robust[column] = (df_robust[column] - df_robust[column].median())  / (df_robust[column].quantile(0.75) - df_robust[column].quantile(0.25))

        df = df_robust
        
        # filter the DataFrame by the selected year range
        filtered_df = df[(df.index.year >= year_range[0]) & (df.index.year <= year_range[1])]
        
        df = filtered_df
        
        fig = make_subplots(rows=len(df.columns), cols=len(df.columns), shared_xaxes=False, shared_yaxes=False)

        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                scatter_trace = go.Scatter(
                    x=df.iloc[:, i],
                    y=df.iloc[:, j],
                    mode='markers',
                    showlegend=False
                )
                hist_trace = go.Histogram(
                    x=df.iloc[:, i],
                    nbinsx=30,
                    showlegend=False
                )
                if i == j:
                    fig.add_trace(hist_trace, row=i+1, col=j+1)
                    fig.update_xaxes(title_text=df.columns[i].replace(' Price Index',''), row=i+1, col=j+1, tickfont=dict(size=7))
                    fig.update_yaxes(title_text=df.columns[j].replace(' Price Index',''), row=i+1, col=j+1, tickfont=dict(size=7))
                else:
                    fig.add_trace(scatter_trace, row=i+1, col=j+1)
                    fig.update_xaxes(title_text=df.columns[i].replace(' Price Index',''), row=i+1, col=j+1, tickfont=dict(size=7))
                    fig.update_yaxes(title_text=df.columns[j].replace(' Price Index',''), row=i+1, col=j+1, tickfont=dict(size=7))

        fig.update_layout(height=1300, width=1300, title='Pair plot of inflation indices')
        
        return fig

    @app.callback(dash.dependencies.Output('time-series-decomposition-plot', 'figure'),
                [dash.dependencies.Input('time-series-decomposition-radio-buttons', 'value'),
                 dash.dependencies.Input('year-range-slider-time-series-decomposition', 'value'), 
                 dash.dependencies.Input('country-selection-dropdown', 'value')])
    def update_time_series_decomposition_plot(selected_index, year_range, value):
        
        country_code = value
        df = readCountryData(country_code)

        df_robust = df.copy()
        # apply robust scaling
        for column in df_robust.columns:
            df_robust[column] = (df_robust[column] - df_robust[column].median())  / (df_robust[column].quantile(0.75) - df_robust[column].quantile(0.25))

        df = df_robust
        
        # filter the DataFrame by the selected year range
        filtered_df = df[(df.index.year >= year_range[0]) & (df.index.year <= year_range[1])]
        
        df = filtered_df
        
        time_series_decomposition = seasonal_decompose(x=df[selected_index], model='additive', period=12)
        trend_decomposition = time_series_decomposition.trend
        seasonal_decomposition = time_series_decomposition.seasonal
        residual_decomposition = time_series_decomposition.resid
        
        fig = make_subplots(rows=4, 
                            cols=1, 
                            shared_xaxes=False, 
                            shared_yaxes=False, 
                            subplot_titles=(
                                selected_index, 
                                selected_index  + ' Trend', 
                                selected_index + ' Seasonal', 
                                selected_index + ' Residual'))
        
        fig.add_trace(go.Scatter(x=df.index, y=df[selected_index], mode='lines', name=selected_index, showlegend=False), row = 1, col = 1)

        fig.add_trace(go.Scatter(x=df.index, y=trend_decomposition, mode='lines', name=selected_index + ' Trend', showlegend=False), row = 2, col = 1)

        fig.add_trace(go.Scatter(x=df.index, y=seasonal_decomposition, mode='lines', name=selected_index + ' Seasonal', showlegend=False), row = 3, col = 1)

        fig.add_trace(go.Scatter(x=df.index, y=residual_decomposition, mode='lines', name=selected_index + ' Residual', showlegend=False), row = 4, col = 1)
        
        fig.update_layout(height=800, width=1300, title='Time Series Decomposition plot of inflation indices')
        
        return fig
    
    @app.callback(dash.dependencies.Output('world-map-month-selected', 'children'),
                dash.dependencies.Input('world-map-month-picker', 'date'))
    def update_selected_month_output(value):
        date = value
        month = pd.to_datetime(date).strftime('%B')
        year = pd.to_datetime(date).year
        return html.H4('Month selected : ' + str(month) + ' ' + str(year),
            style={'textAlign': 'center',
                    'color': '#000205'}
            )

    @app.callback(
        dash.dependencies.Output('world-map-inflation-index', 'figure'),
        [dash.dependencies.Input('world-map-month-picker', 'date'),
         dash.dependencies.Input('world-map-radio-buttons', 'value')]
    )
    def update_world_map(date, selected_index):
        
        month = str(pd.to_datetime(date).strftime("%m"))
        year = str(pd.to_datetime(date).year)
        
        column_name = str(year) + str(month)
        
        if selected_index == 'Official Core Consumer Price Index':
            ccpiBlob = bucket.blob('ccpi_m.csv')
            ccpiData = pd.read_csv(ccpiBlob.download_as_string())
            ccpiData = ccpiData[['Country Code', 'Country', column_name]]
            df = ccpiData
        elif selected_index == 'Energy Price Index':
            ecpiBlob = bucket.blob('ecpi_m.csv')
            ecpiData = pd.read_csv(ecpiBlob.download_as_string())
            ecpiData = ecpiData[['Country Code', 'Country', column_name]]
            df = ecpiData
        elif selected_index == 'Food Price Index':
            fcpiBlob = bucket.blob('fcpi_m.csv')
            fcpiData = pd.read_csv(fcpiBlob.download_as_string())
            fcpiData = fcpiData[['Country Code', 'Country', column_name]]
            df = fcpiData
        elif selected_index == 'Headline Consumer Price Index':
            hcpiBlob = bucket.blob('hcpi_m.csv')
            hcpiData = pd.read_csv(hcpiBlob.download_as_string())
            hcpiData = hcpiData[['Country Code', 'Country', column_name]]
            df = hcpiData
        else:
            ppiBlob = bucket.blob('ppi_m.csv')
            ppiData = pd.read_csv(ppiBlob.download_as_string())
            ppiData = ppiData[['Country Code', 'Country', column_name]]
            df = ppiData
            
        df[column_name] = np.log(df[column_name])
               
        fig = px.choropleth(df, locations='Country', 
                            locationmode='country names', 
                            color=column_name,
                            color_continuous_scale='Viridis',
                            range_color=[df[column_name].min(), df[column_name].max()],
                            color_continuous_midpoint=df[column_name].median(),
                            projection='orthographic')
        
        #fig.update_layout(autosize=True)
        fig.update_layout(height=700, width=1200, autosize=True)
        
        return fig
    
    @app.callback(
        dash.dependencies.Output('prediction-months-selected', 'children'),
        dash.dependencies.Input('prediction-month-picker', 'start_date'),
        dash.dependencies.Input('prediction-month-picker', 'end_date'),
        dash.dependencies.Input('precision-selection-dropdown','value')
    )
    def update_prediction_months_output(start_date, end_date, precision):
        
        start_month = pd.to_datetime(start_date).strftime('%B')
        start_year = pd.to_datetime(start_date).year

        end_month = pd.to_datetime(end_date).strftime('%B')
        end_year = pd.to_datetime(end_date).year
        return html.H4('Start Month : ' + str(start_month) + ' ' + str(start_year) + ', End Month : ' + str(end_month) + ' ' + str(end_year) + ', Precision : ' + precision,
            style={'textAlign': 'center',
                    'color': '#000205'}
            )
    
    @app.callback(
        dash.dependencies.Output('inflation-index-prediction', 'figure'),
        [dash.dependencies.Input('prediction-month-picker', 'start_date'),
        dash.dependencies.Input('prediction-month-picker', 'end_date'),
        dash.dependencies.Input('prediction-radio-buttons', 'value'),
        dash.dependencies.Input('precision-selection-dropdown','value'),
        dash.dependencies.Input('country-selection-dropdown', 'value')]
    )
    def update_prediction_months_output(start_date, end_date, index, precision, country_code):

        df = readCountryData(country_code)

        # select the 'values' column
        values_col = df[index]

        # calculate the median and quartiles of the column
        median = values_col.median()
        q25 = values_col.quantile(0.25)
        q75 = values_col.quantile(0.75)

        # apply the formula to normalize the values
        values_col = (values_col - median) / (q75 - q25)

        # update the column in the original dataframe
        df[index] = values_col

        test_size = 12

        #test sets
        lstmTestDF = df.tail(12)

        # train sets
        # get the number of rows in the dataframe
        num_rows = df.shape[0]

        # get all rows except the last 12 rows
        lstmTrainDF = df.iloc[:num_rows-12, :]

        X_train_dict = {}
        y_train_dict = {}
        X_test_dict = {}
        y_test_dict = {}
        window = 3

        col = index

        X_train_dict[col], y_train_dict[col] = segregate_train_test_data(data=lstmTrainDF[col], window_size=window)
        X_test_dict[col], y_test_dict[col] = segregate_train_test_data(data=lstmTestDF[col], window_size=window)
        X_train_dict[col] = np.reshape(X_train_dict[col], (X_train_dict[col].shape[0], X_train_dict[col].shape[1], 1))
        X_test_dict[col] = np.reshape(X_test_dict[col], (X_test_dict[col].shape[0], X_test_dict[col].shape[1], 1))
            
        X_dict = {}
        y_dict = {}
  
        X_dict[col], y_dict[col] = segregate_train_test_data(data=lstmTrainDF[col], window_size=3)
        X_dict[col] = np.reshape(X_dict[col], (X_dict[col].shape[0], X_dict[col].shape[1], 1))   
        
        learning_rate = result_dict[col]['learning_rate']
        epochs = result_dict[col]['epochs']

        def create_lstm_model_implementation(col, learning_rate, epochs):
            model = Sequential()
            model.add(LSTM(64, input_shape=(X_train_dict[col].shape[1], X_train_dict[col].shape[2])))
            model.add(Dense(1))
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(loss='mse', optimizer=optimizer)
            model.fit(X_train_dict[col], y_train_dict[col], epochs=epochs, batch_size=64, validation_data=(X_test_dict[col], y_test_dict[col]))
            return model

        if (precision == 'low'):
            model = create_lstm_model_implementation(col= col,learning_rate=learning_rate ,epochs=epochs)
        elif (precision == 'medium'):
            model = create_lstm_model_implementation(col= col,learning_rate=learning_rate / 5 ,epochs=epochs * 5)
        else:
            model = create_lstm_model_implementation(col= col,learning_rate=learning_rate / 10 ,epochs=epochs * 10)

        predictions = model.predict(X_test_dict[col])

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) 

        # make predictions for future values
        future_dates = pd.date_range(start_date,end_date, freq='MS')
        future_values = []
        last_data = df[[col]].tail(3)

        for i in range(len(future_dates)):
            X = np.expand_dims(last_data, axis=0)
            y_pred = model.predict(X)[0][0]
            future_values.append(y_pred)
            last_data = np.vstack((last_data[1:], [y_pred]))

        future_data = pd.DataFrame(future_values, columns= ['prediction'], index=future_dates)

        actual_trace = go.Scatter(x=df.index, y=df[col], mode='lines', name='Actual')
        prediction_trace = go.Scatter(x=future_data.index, y=future_data['prediction'], mode='lines', name='Prediction')

        layout = go.Layout(title='Inflation indices prediction',
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Value'))

        fig = go.Figure(data=[actual_trace, prediction_trace], layout=layout)


        return fig

    app.layout = html.Div([
    html.H1(children='Current Topics In Data Science',
            style={'textAlign': 'center',
                    'color': '#000205'}
            ),
    html.Br(),
    countryCodeDropdown,
    displayCountryName,
    html.Hr(),
    html.H4(children='Select year range to view Line Plot plots for Inflation indices',
            style={'textAlign': 'center',
                    'color': '#000205'}
            ),
    dcc.RangeSlider(
        id='year-range-slider-line-plot',
        min=structureDF.index.year.min(),
        max=structureDF.index.year.max(),
        step=1,
        marks={year: str(year) for year in structureDF.index.year.unique()},
        value=[structureDF.index.year.min(), structureDF.index.year.max()]
    ),
    html.Div([
    dcc.Loading(id="loading-icon-line-plot", children=[
        dcc.Graph(id='line-plot')
    ], type="circle")
    ]),
    #dcc.Graph(id='line-plot'),
    html.Hr(),
    html.H4(children='Select year range to view Histogram plots for Inflation indices',
            style={'textAlign': 'center',
                    'color': '#000205'}
            ),
    dcc.RangeSlider(
        id='year-range-slider-histogram',
        min=structureDF.index.year.min(),
        max=structureDF.index.year.max(),
        step=1,
        marks={year: str(year) for year in structureDF.index.year.unique()},
        value=[structureDF.index.year.min(), structureDF.index.year.max()]
    ),
    html.Div([
    dcc.Loading(id="loading-icon-histogram", children=[
        dcc.Graph(id='histogram')
    ], type="circle")
    ]),
    #dcc.Graph(id='histogram'),
    html.Hr(),
    html.H4(children='Select year range to view violin plots for Inflation indices',
            style={'textAlign': 'center',
                    'color': '#000205'}
            ),
    dcc.RangeSlider(
        id='year-range-slider-violin',
        min=structureDF.index.year.min(),
        max=structureDF.index.year.max(),
        step=1,
        marks={year: str(year) for year in structureDF.index.year.unique()},
        value=[structureDF.index.year.min(), structureDF.index.year.max()]
    ),
    html.Div([
    dcc.Loading(id="loading-icon-inflation-violin-plot", children=[
        dcc.Graph(id='inflation-violin-plot')
    ], type="circle")
    ]),
    #dcc.Graph(id='inflation-violin-plot'),
    html.Hr(),
    html.H4(children='Select year range to view pair plots for Inflation indices',
            style={'textAlign': 'center',
                    'color': '#000205'}
            ),
    dcc.RangeSlider(
        id='year-range-slider-pair-plot',
        min=structureDF.index.year.min(),
        max=structureDF.index.year.max(),
        step=1,
        marks={year: str(year) for year in structureDF.index.year.unique()},
        value=[structureDF.index.year.min(), structureDF.index.year.max()]
    ),
    html.Div([
    dcc.Loading(id="loading-icon-pair-plot", children=[
        dcc.Graph(id='pair-plot')
    ], type="circle")
    ]),
    #dcc.Graph(id='pair-plot'),
    html.Hr(),
    html.Div([
            html.H4('Select index to Display:'),
            dcc.RadioItems(
                id='time-series-decomposition-radio-buttons',
                options=[
                    {'label': 'Official Core Consumer Price Index', 'value': 'Official Core Consumer Price Index'},
                    {'label': 'Energy Price Index', 'value': 'Energy Price Index'},
                    {'label': 'Food Price Index', 'value': 'Food Price Index'},
                    {'label': 'Headline Consumer Price Index', 'value': 'Headline Consumer Price Index'},
                    {'label': 'Producer Price Index', 'value': 'Producer Price Index'},
                ],
                value='Official Core Consumer Price Index'
            ),
        ], style={'textAlign': 'center'}),
    html.H4(children='Select year range to view time-series decomposition plots for Inflation indices',
            style={'textAlign': 'center',
                    'color': '#000205'}
            ),
    dcc.RangeSlider(
        id='year-range-slider-time-series-decomposition',
        min=structureDF.index.year.min(),
        max=structureDF.index.year.max(),
        step=1,
        marks={year: str(year) for year in structureDF.index.year.unique()},
        value=[structureDF.index.year.min(), structureDF.index.year.max()]
    ),
    html.Div([
    dcc.Loading(id="loading-icon-time-series-decomposition-plot", children=[
        dcc.Graph(id='time-series-decomposition-plot')
    ], type="circle")
    ]),
    #dcc.Graph(id='time-series-decomposition-plot'),
    html.Hr(),
    html.H4(children='Select a month to view Inflation indices for available countries',
            style={'textAlign': 'center',
                    'color': '#000205'}
            ),
    html.Div(
        dcc.DatePickerSingle(
            id='world-map-month-picker',
            date=structureDF.index.max(),
            clearable=True,
            with_portal=True,
            min_date_allowed=structureDF.index.min(),
            max_date_allowed=structureDF.index.max(),
            display_format='MMM Y'
        ), style={'textAlign': 'center',
                    'color': '#000205'}
    ),
    html.Div([
            html.H4('Select index to Display:'),
            dcc.RadioItems(
                id='world-map-radio-buttons',
                options=[
                    {'label': 'Official Core Consumer Price Index', 'value': 'Official Core Consumer Price Index'},
                    {'label': 'Energy Price Index', 'value': 'Energy Price Index'},
                    {'label': 'Food Price Index', 'value': 'Food Price Index'},
                    {'label': 'Headline Consumer Price Index', 'value': 'Headline Consumer Price Index'},
                    {'label': 'Producer Price Index', 'value': 'Producer Price Index'},
                ],
                value='Official Core Consumer Price Index'
            ),
        ], style={'textAlign': 'center'}),
    displaySelectedMonth,
    html.Div([
    dcc.Loading(id="loading-icon-world-map-inflation-index", children=[
        dcc.Graph(id='world-map-inflation-index')
    ], type="circle")
    ]),
    #dcc.Graph(id='world-map-inflation-index'),
    html.Hr(),
    html.H3(children='Inflation predictions',
            style={'textAlign': 'center',
                    'color': '#000205'}
            ),
    html.Div([
        html.H4('Select index to Display:'),
        dcc.RadioItems(
            id='prediction-radio-buttons',
            options=[
                {'label': 'Official Core Consumer Price Index', 'value': 'Official Core Consumer Price Index'},
                {'label': 'Energy Price Index', 'value': 'Energy Price Index'},
                {'label': 'Food Price Index', 'value': 'Food Price Index'},
                {'label': 'Headline Consumer Price Index', 'value': 'Headline Consumer Price Index'},
                {'label': 'Producer Price Index', 'value': 'Producer Price Index'},
            ],
            value='Official Core Consumer Price Index'
        ),
    ], style={'textAlign': 'center'}),
    html.Div([
        html.H4('Select Start and End months:'),
        dcc.DatePickerRange(
            id='prediction-month-picker',
            start_date_placeholder_text="Start Period",
            end_date_placeholder_text="End Period",
            clearable=True,
            with_portal=True,
            min_date_allowed=structureDF.index.max(),
            max_date_allowed=structureDF.index.max()+ timedelta(days=5*365),
            display_format='MMM Y'
    )], style={'textAlign': 'center',
                    'color': '#000205'}
    ),
    html.H4(children='Select precision level:',
                style={'textAlign': 'center'}
        ),
    precisionDropdown,
    displayPredictionMonths,
    dcc.Loading(
        id="loading-inflation-index-prediction",
        children=dcc.Graph(
            id='inflation-index-prediction',
            figure={}
        ),
        type="circle",
    ),
    html.Hr()
    ])
    app.run(host='0.0.0.0', port=8080, debug=True)