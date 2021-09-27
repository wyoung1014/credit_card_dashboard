import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

url_cc = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'


df = pd.read_csv('default of credit card clients.csv', skiprows = [1])

#recode x3
df['X3'] = df['X3'].replace([4,5,6],[0,0,0]) 

#Set names of catagorial variables
df_recoded= df.copy(deep=True)
paycode  = { -2 :'No consumption', 0 :'Revolving Credit', -1 :'on time',1:'delay_1',2:'delay_2',3:'delay_3',4:'delay_4',5:'delay_5',6:'delay_6',7:'delay_7',8:'delay_8',9:'delay_9+'}
edu  = {1:'graduate school',2:'university',3:'high school',0:'other'}
mar  ={1:'married',2:'single',3:'divorced', 0:'other'}
sex = {1:'male',2:'female'}

df_recoded[['X6','X7','X8','X9','X10','X11']] = df[['X6','X7','X8','X9','X10','X11']].replace(paycode)
df_recoded['X3'] = df['X3'].replace(edu)
df_recoded['X4'] = df['X4'].replace(mar)
df_recoded['X2'] = df['X2'].replace(sex)

#split x and y
xy = pd.get_dummies(df_recoded, columns=['X2','X3','X4','X6','X7','X8','X9','X10','X11'], drop_first=False)
x, y = xy.drop(xy.columns[[3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,27,38,49,60,71,81]], axis=1), xy['Y']

#splt properly reduced data
X_train, X_test = train_test_split(x, test_size=0.33, random_state=341343)
y_train, y_test = train_test_split(y, test_size=0.33, random_state=341343)

# open model
#with open('LogReg2.pickle', 'rb') as f:
   # LogReg2 = pickle.load(f)
LogReg2  =LogisticRegression(max_iter = 5000, random_state=35475635, C=10000, penalty='l1', solver='liblinear', verbose = True)   
LogReg2.fit(X_train, y_train)
# Define Bond Rating Fomula
def get_BR(prob):
    if 0 <= prob <=.1:
        return 'A'
    elif .10 <= prob <= .20:
        return 'B'
    elif .20 <= prob <= .30:
        return 'C'
    elif prob >.30:
        return 'D'



#use model to add columns
Pd = confusion_matrix(y_test, LogReg2.predict(X_test), normalize='all')[1,1]/np.sum (confusion_matrix(y_test, LogReg2.predict(X_test), normalize='all'), axis=0)[1]
d= LogReg2.predict_proba(x)[:,1] * Pd

x['Prob']=d
x['Rating'] = x['Prob'].apply(get_BR).astype('category') 
df_recoded['Rating'], df_recoded['Prob'] = x['Rating'], x['Prob']







app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]) # what do meta tags do?

app.layout = html.Div([  #row 1
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('WY.svg'),
                     id='WY',
                     style={
                         "height": "60px",
                         "width": "auto",
                         "margin-bottom": "25px",
                     },
                     )
        ],
            className="one-third column", #class names are from CSS
        ),
        html.Div([
            html.Div([
                html.H3("Credit Card Risk Dashboard", style={"margin-bottom": "0px", 'color': 'white'}),
                html.H5("By Wesley James Young", style={"margin-top": "0px", 'color': 'white'}),
            ])
        ], className="one-half column", id="title"), # ID Title in CSS

        html.Div([
            html.H6('Draft Item, not final work product',
                    style={'color': 'orange'}),

        ], className="one-third column", id='title1'),

    ], id="header", className="row flex-display", style={"margin-bottom": "25px"}), # header and flex display mentioned in CSS


    html.Div([                          #row 3
        html.Div([

                    html.P('Select Chart:', className='fix_label',  style={'color': 'white'}),

                     dcc.Dropdown(id='w_charts', #goes to callback 1
                                  multi=False,
                                  clearable=True,
                                  value= None,
                                  placeholder='Select Filter',
                                  options=[{'label': 'Sex', 'value': 'X2'},
                                           {'label': 'Education', 'value': 'X3'},
                                           {'label': 'Marital status', 'value': 'X4'}
                                           ]
                 ),

        ], className="create_container three columns", id="cross-filter-options"),
            html.Div([
                      dcc.Graph(id='fig',  # callback 5
                              config={'displayModeBar': 'hover'}),
                              ], className="create_container four columns")]),
                html.Div([
                      dcc.Graph(id='box',  # callback 5
                              config={'displayModeBar': 'hover'}),
                              ], className="create_container four columns"),
                html.Div([
                      dcc.Graph(id='fig3',  # callback 5
                              config={'displayModeBar': 'hover'}),
                              ], className="create_container four columns")

    ])
# Create pie chart (total casualties)
@app.callback(Output('box', 'figure'),
              Output('fig', 'figure'),
              Output('fig3', 'figure'),
              [Input('w_charts', 'value')])

def update_graph(w_charts):
    fig = px.box(df_recoded, x = 'Rating', y = 'X1', color= w_charts,
                            labels={
                     "X1": "Balance Limit"
                 },  
              category_orders={"Rating": ["A", "B", "C", "D"]},
             title="Bond Ratings and Balance Limits")
    fig2 = px.histogram(df_recoded['Rating'],  
                      labels={
                     "value": "Rating",
                          'probability density':'Proportion of Bonds Classes'
                 },  
                    category_orders={"value": ["A", "B", "C", "D"]},
                    title="Rating Distibution", 
                   histnorm='probability density' )
    fig3 = px.box(df_recoded, x = w_charts, y = 'Prob',
                            labels={
                     "Prob": "Probability of Default"
                 },  
              category_orders={"Rating": ["A", "B", "C", "D"]},
             title="Probabilities of Default")
    return fig, fig2, fig3



if __name__ == '__main__':
    app.run_server(debug=False)
