import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, dash_table
import plotly.express as px
import geopandas as gpd
from shapely import wkt
import json
import pickle

from sklearn.ensemble import RandomForestClassifier  # This should work now
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
df = pd.read_csv('subsidie.csv')
df = df[df.Bedragaangevraagd != 0]
df = df.drop(columns=['Publicatiedatumverleningsbesluit', 'Publicatiedatumvaststellingsbesluit'])
df = df.dropna(subset=['Bedragaangevraagd', 'Bedragverleend', 'Bedragvastgesteld'])

# Filter stadsdeel data
stadsdeel_data = df[df['Organisatieonderdeel'].str.contains('stadsdeel', case=False, na=False) &
                    ~df['Organisatieonderdeel'].str.contains('Stadsdeeloverstijgend', case=False, na=False)]
stadsdeel_data['Organisatieonderdeel'] = stadsdeel_data['Organisatieonderdeel'].str.replace('Stadsdeel ', '',
                                                                                            case=False)

# Load geographic data
geo_stadsdelen_df = pd.read_excel('stadsdelen.xlsx')
geo_stadsdelen_df = geo_stadsdelen_df[~geo_stadsdelen_df['Stadsdeel'].isin(['Weesp', 'Westpoort'])]
merged_geo_df = pd.merge(geo_stadsdelen_df, stadsdeel_data, left_on='Stadsdeel', right_on='Organisatieonderdeel')

grouped_df = merged_geo_df.groupby(['Subsidiejaar', 'Stadsdeel']).agg({
    'Bedragverleend': 'sum',
    'WKT_LNG_LAT': 'first'
}).reset_index()

# Calculate total per year
df_sub_per_year = df.groupby("Subsidiejaar").sum()
float_cols = df_sub_per_year.select_dtypes(include=['float']).columns
df_sub_per_year[float_cols] = df_sub_per_year[float_cols].astype(int)
df_sub_per_year['Percentage_verleend'] = (
            df_sub_per_year['Bedragverleend'] / df_sub_per_year['Bedragaangevraagd'] * 100)

# Get unique years
unique_years = sorted(df['Subsidiejaar'].unique())
year_options = [{'label': str(year), 'value': year} for year in unique_years]
year_options.insert(0, {'label': 'Alle Jaren', 'value': 'all'})

# Load the machine learning model
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Helper functions
def format_large_number(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} billion"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} million"
    elif number >= 1_000:
        return f"{number / 1_000:.2f} thousand"
    else:
        return str(number)
def create_card(p_text, h4_text, color, textcolor):
    return dbc.Card([
        dbc.CardBody([
            html.P(p_text, className="card-value", style={
                'margin': '0px',
                'fontSize': '22px',
                'fontWeight': 'bold',
                'color': textcolor
            }),
            html.H4(h4_text, className="card-title", style={
                'margin': '0px',
                'fontSize': '18px',
                'fontWeight': 'bold',
                'color': textcolor
            })
        ], style={'textAlign': 'center'}),
    ], style={
        'paddingBlock': '10px',
        'backgroundColor': color,
        'border': 'none',
        'borderRadius': '10px'
    })


# Initialize Dash app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.Button("Home", href="/home", color="light")),
        dbc.NavItem(dbc.Button("Discover Year", href="/", color="light", style={"margin-left": "10px"})),
        dbc.NavItem(dbc.Button("Subsidy Predictor", href="/predict", color="light", style={"margin-left": "10px"})),
        dbc.NavItem(dbc.Button("Explore", href="/explore", color="light", style={"margin-left": "10px"}))
    ],
    brand='Amsterdam Subsidy Dashboard',
    brand_href="#",
    color="#ec0000",
    dark=True,
    fluid=True
)

# Year selector card
card_year = dbc.Card([
    dbc.CardBody([
        html.P("Select Year", className="card-value", style={
            'margin': '0px', 'fontSize': '20px', 'fontWeight': 'bold', 'color': '#FFFFFF'
        }),
        dcc.Dropdown(
            id='year-selector',
            options=year_options,
            value='all',
            clearable=False,
            style={'marginTop': '5px'}
        )
    ], style={'textAlign': 'center'}),
], style={
    'paddingBlock': '2px',
    "backgroundColor": '#ec0000',
    'border': 'none',
    'borderRadius': '10px'
})
# Homepage layout
home_layout = dbc.Container([
    dbc.Row([
        dbc.Col(
dbc.Card([
        dbc.CardBody([
            html.P('Welcome to the Amsterdam Subsidy Explorer!',
                   className="card-value", style={
                'margin': '0px',
                'fontSize': '22px',
                'fontWeight': 'bold',
                'color': '#000000',
                'textAlign': 'center',
                'margin-bottom': '20px'
            }),
            html.H4('Our platform is designed to bring transparency to how public funds are distributed across various sectors, districts, and projects in Amsterdam. '
                    'Whether you’re a citizen, business,'
                    ' or organization, you can easily access and analyze detailed information'
                    ' about subsidies—helping you stay informed and make data-driven decisions.',
                    className="card-title", style={
                'margin': '0px',
                'fontSize': '18px',
                'fontWeight': 'normal',
                'color': '#000000',
                'textAlign': 'start'
            }),
            html.H4('Here’s how you can get started:',
                    className="card-title", style={
                'margin': '0px',
                'fontSize': '18px',
                'fontWeight': 'normal',
                'color': '#000000',
                'textAlign': 'start',
                'margin-top': '20px'
            }),
            html.H4('Filter the Data by Year: Use our filtering tool to view subsidy data from specific years,'
                    ' helping you track how public funding has changed over time.',
                    className="card-title", style={
                'margin': '0px',
                'fontSize': '18px',
                'fontWeight': 'normal',
                'color': '#000000',
                'textAlign': 'start',
                'margin-top': '10px'
            }),
            html.H4('Subsidy Predictor: Curious about your chances of receiving a subsidy? '
                    'Enter your project details into our subsidy predictor to estimate whether your requested amount '
                    'will likely be fully granted, partially approved, or rejected.',
                    className="card-title", style={
                'margin': '0px',
                'fontSize': '18px',
                'fontWeight': 'normal',
                'color': '#000000',
                'textAlign': 'start',
                'margin-top': '10px'
            }),
            html.H4('Explore the Data: Dive into the data directly by viewing a detailed table of subsidy requests and grants.',
                    className="card-title", style={
                'margin': '0px',
                'fontSize': '18px',
                'fontWeight': 'normal',
                'color': '#000000',
                'textAlign': 'start',
                'margin-top': '10px'
            }),
            html.H4('With our user-friendly interface, you can navigate through the subsidy data with ease. Let’s start exploring!',
                    className="card-title", style={
                'margin': '0px',
                'fontSize': '18px',
                'fontWeight': 'normal',
                'color': '#000000',
                'textAlign': 'start',
                'margin-top': '20px'
            }),

        ]),
    ], style={
        'paddingBlock': '10px',
        'backgroundColor': '#D3D3D3',
        'border': 'none',
        'borderRadius': '10px',
        'margin-top': '30px'
    })


        ) ])
])

# Dashboard layout
dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div(card_year), width=3),
        dbc.Col(html.Div(id='card-1-request'), width=3),
        dbc.Col(html.Div(id='card-2-approval'), width=3),
        dbc.Col(html.Div(id='card-3-grant'), width=3)],
        align="start", className="mb-4"),
    dbc.Row([
        dbc.Col([html.Div(id='pie_title'),
                dcc.Graph(id="pie_right")], width=6),
        dbc.Col([html.Div(id='table_title'),
                html.Div(id="table")], width=6),
    ], align="start", className="mb-4"),
    dbc.Row([
        dbc.Col([html.Div(id='map_title'),
                 dcc.Graph(id="choropleth-map")], width=8),
        dbc.Col(['Bar Chart'], width=4),
    ]),
], className="p-5")

# Prediction layout
prediction_layout = dbc.Container([
    html.H1("Subsidy Prediction", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            dbc.Form([
                dbc.Row([
                    dbc.Label("Amount Requested (€)", html_for="amount-input", width=4),
                    dbc.Col(
                        dbc.Input(type="number", id="amount-input", placeholder="Enter amount"),
                        width=8,
                    ),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Policy Area", html_for="policy-dropdown", width=4),
                    dbc.Col(
                        dcc.Dropdown(id="policy-dropdown", options=[
                            {'label': 'Algemeen', 'value': 0},
                            {'label': 'Bedrijfsvoering', 'value': 1},
                            {'label': 'Bestuur en ondersteuning', 'value': 2},
                            {'label': 'Cultuur', 'value': 3},
                            {'label': 'Dienstverlening en informatie', 'value': 4},
                            {'label': 'Diversiteit', 'value': 5},
                            {'label': 'Economie', 'value': 6},
                            {'label': 'Inkomen', 'value': 7},
                            {'label': 'Jeugd', 'value': 8},
                            {'label': 'Monumenten', 'value': 9},
                            {'label': 'Onderwijs', 'value': 10},
                            {'label': 'Openbare orde en veiligheid', 'value': 11},
                            {'label': 'Ruimte en duurzaamheid', 'value': 12},
                            {'label': 'Sociaal', 'value': 13},
                            {'label': 'Sociale basis', 'value': 14},
                            {'label': 'Sport', 'value': 15},
                            {'label': 'Stedelijke ontwikkeling en wonen', 'value': 16},
                            {'label': 'Verkeer en openbare ruimte', 'value': 17},
                            {'label': 'Werk', 'value': 18},
                            {'label': 'Zorg', 'value': 19}
                        ]),
                        width=8,
                    ),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Organizational Unit", html_for="org-dropdown", width=4),
                    dbc.Col(
                        dcc.Dropdown(id="org-dropdown", options=[
                            {'label': 'Bestuur en Organisatie', 'value': 0},
                            {'label': 'Communicatie', 'value': 1},
                            {'label': 'Economische Zaken', 'value': 2},
                            {'label': 'Grond en Ontwikkeling', 'value': 3},
                            {'label': 'Inkomen', 'value': 4},
                            {'label': 'Kunst en Cultuur', 'value': 5},
                            {'label': 'Maatschappelijke Voorzieningen', 'value': 6},
                            {'label': 'Monumenten en Archeologie', 'value': 7},
                            {'label': 'Onderwijs', 'value': 8},
                            {'label': 'Jeugd en Zorg', 'value': 9},
                            {'label': 'Participatie', 'value': 10},
                            {'label': 'Ruimte en Duurzaamheid', 'value': 11},
                            {'label': 'Sport en Bos', 'value': 12},
                            {'label': 'Stadsbeheer en Gebiedsgericht werken', 'value': 13},
                            {'label': 'Stadsdeel Centrum', 'value': 14},
                            {'label': 'Stadsdeel Nieuw-West', 'value': 15},
                            {'label': 'Stadsdeel Noord', 'value': 16},
                            {'label': 'Stadsdeel Oost', 'value': 17},
                            {'label': 'Stadsdeel West', 'value': 18},
                            {'label': 'Stadsdeel Zuid', 'value': 19},
                            {'label': 'Stadsdeel Zuidoost', 'value': 20},
                            {'label': 'Stadsdeel Overstijgend', 'value': 21},
                            {'label': 'Stadsgebied Weesp', 'value': 22},
                            {'label': 'Verkeer en openbare ruimte', 'value': 23},
                            {'label': 'Werk', 'value': 24},
                            {'label': 'Wonen', 'value': 25},
                            {'label': 'Zuidas', 'value': 26}
                        ]),
                        width=8,
                    ),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Periodicity Type", html_for="periodicity-dropdown", width=4),
                    dbc.Col(
                        dcc.Dropdown(id="periodicity-dropdown", options=[
                            {"label": "One-time", "value": 0},
                            {"label": "Book year", "value": 1}
                        ]),
                        width=8,
                    ),
                ], className="mb-3"),
                dbc.Button("Predict", color="primary", id="predict-button", className="mt-3"),
            ]),
        ], md=6),
        dbc.Col([
            html.Div(id="prediction-output", className="mt-4")
        ], md=6),
    ]),
])

# Explore data layout
explore_layout = dbc.Container(
    dbc.Row([
        dbc.Col(
            html.Div([
                dash_table.DataTable(
                    id='datatable-interactivity',
                    columns=[
                        {"name": i, "id": i}
                        if i == "iso_alpha3" or i == "year" or i == "id"
                        else {"name": i, "id": i}
                        for i in df.columns
                    ],
                    data=df.to_dict('records'),  # the contents of the table
                    editable=False,  # allow editing of data inside all cells
                    filter_action="native",  # allow filtering of data by user ('native') or not ('none')
                    sort_action="native",  # enables data to be sorted per-column by user or not ('none')
                    sort_mode="single",  # sort across 'multi' or 'single' columns
                    selected_columns=[],  # ids of columns that user selects
                    selected_rows=[],  # indices of rows that user selects
                    page_action="native",  # all data is passed to the table up-front or not ('none')
                    page_current=0,  # page number that user is on
                    page_size=10,  # number of rows visible per page
                    style_cell={  # ensure adequate header width when text is shorter than cell's text
                        'minWidth': 95, 'maxWidth': 260, 'width': 95
                    },
                    style_cell_conditional=[  # align text columns to left. By default they are aligned to right
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in ['country', 'iso_alpha3']
                    ],
                    style_data={  # overflow cells' content into multiple lines
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    }
                )

            ])

        )
    ], style={'margin-top': '30px'}), fluid=True
)

app.config.suppress_callback_exceptions = True
# Main app layout
app.layout = html.Div([
    dcc.Location(id="url"),
    navbar,
    html.Div(id='page-content')

])

# Callbacks
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/predict':
        return prediction_layout
    if pathname == '/home':
        return home_layout
    if pathname == '/explore':
        return explore_layout
    else:
        return dashboard_layout


@app.callback(
    [Output('card-1-request', 'children'),
     Output('card-2-approval', 'children'),
     Output('card-3-grant', 'children')],
    [Input('year-selector', 'value')]
)
def update_cards(selected_year):
    if selected_year == 'all':
        value_request = df_sub_per_year["Bedragaangevraagd"].sum()
        value_grant = df_sub_per_year["Bedragverleend"].sum()
        value_approval = round(((value_grant / value_request) * 100), 2)
    else:
        value_request = df_sub_per_year.loc[df_sub_per_year.index == selected_year, 'Bedragaangevraagd'].values[0]
        value_grant = df_sub_per_year.loc[df_sub_per_year.index == selected_year, 'Bedragverleend'].values[0]
        value_approval = round(
            df_sub_per_year.loc[df_sub_per_year.index == selected_year, 'Percentage_verleend'].values[0], 2)

    return (create_card("Total Requested", str(format_large_number(value_request)) + " €", '#ec0000', '#FFFFFF'),
            create_card("Approval Rate", str(format_large_number(value_approval)) + "%", '#ec0000', '#FFFFFF'),
            create_card("Total Granted", str(format_large_number(value_grant)) + " €", '#ec0000', '#FFFFFF'))


@app.callback(
    Output('table', 'children'),
    [Input('year-selector', 'value')]
)
def create_top_5_table(selected_year):
    if selected_year != 'all':
        df_filtered = df[df['Subsidiejaar'] == selected_year]
    else:
        df_filtered = df

    df_grouped = df_filtered.groupby(['Aanvrager']).agg(
        total_bedragverleend=pd.NamedAgg(column='Bedragverleend', aggfunc='sum'),
        aanvraag_count=pd.NamedAgg(column='Dossiernummer', aggfunc='count')
    ).reset_index()

    df_top5 = df_grouped.sort_values(by='total_bedragverleend', ascending=False).head(6)
    df_top5['total_bedragverleend'] = df_top5['total_bedragverleend'].apply(format_large_number)
    table = dbc.Table.from_dataframe(df_top5, striped=True, bordered=True, hover=True)

    return table


@app.callback(
    Output('pie_right', 'figure'),
    [Input('year-selector', 'value')]
)
def plot_pie_chart_bedragverleend(selected_year):
    if selected_year != 'all':
        df_filtered = df[df['Subsidiejaar'] == selected_year]
    else:
        df_filtered = df

    df_grouped = df_filtered.groupby('Beleidsterrein').agg(
        total_bedragverleend=pd.NamedAgg(column='Bedragverleend', aggfunc='sum')
    ).reset_index()

    fig = px.pie(df_grouped, values='total_bedragverleend', names='Beleidsterrein',
                 labels={'total_bedragverleend': 'Amount Granted'})

    return fig


# Callback to update the title for Plot X dynamically
@app.callback(
    Output('pie_title', 'children'),
    Input('year-selector', 'value')
)
def update_barplot_title(selected_year):
    # Generate the dynamic title text
    title_text = f'Barplot of Year: {selected_year if selected_year != "all" else "All Time"}'

    # Return title with question mark and tooltip
    return html.Div([
        html.Span(title_text, style={
            'fontSize': '22px',
            'fontWeight': 'bold',
            'color': '#000000',
            'text-align': 'center'
        }),

        # The question mark icon with a tooltip
        html.Span(
            "?",
            id="tooltip-target",
            style={
                "cursor": "pointer",
                "color": "#ec0000",
                "fontWeight": "bold",
                "marginLeft": "10px",  # Space between title and question mark
                "fontSize": "20px",  # Make the question mark bigger
                "border": "2px solid #ec0000",  # Circular border
                "borderRadius": "50%",  # Make the border circular
                "padding": "0",  # Remove extra padding
                "width": "30px",  # Set fixed width
                "height": "30px",  # Set fixed height
                "lineHeight": "30px",  # Align text vertically
                "display": "inline-block",  # Ensure inline behavior
                "textAlign": "center"  # Center the "?" horizontally
            }
        ),

        # Tooltip for the question mark
        dbc.Tooltip(
            "This plot shows the top requesters for the selected year. Hover over the plot to explore details.",
            target="tooltip-target",
            placement="top",
        )
    ], style={'text-align': 'center'})  # Center the entire title block


# Callback to update the title for Plot X dynamically
@app.callback(
    Output('table_title', 'children'),
    Input('year-selector', 'value')
)
def update_tavle_title(selected_year):
    # Generate the dynamic title text
    title_text = f'Table of Year: {selected_year if selected_year != "all" else "All Time"}'

    # Return title with question mark and tooltip
    return html.Div([
        html.Span(title_text, style={
            'fontSize': '22px',
            'fontWeight': 'bold',
            'color': '#000000',
            'text-align': 'center'
        }),

        # The question mark icon with a tooltip
        html.Span(
            "?",
            id="tooltip-target",
            style={
                "cursor": "pointer",
                "color": "#ec0000",
                "fontWeight": "bold",
                "marginLeft": "10px",  # Space between title and question mark
                "fontSize": "20px",  # Make the question mark bigger
                "border": "2px solid #ec0000",  # Circular border
                "borderRadius": "50%",  # Make the border circular
                "padding": "0",  # Remove extra padding
                "width": "30px",  # Set fixed width
                "height": "30px",  # Set fixed height
                "lineHeight": "30px",  # Align text vertically
                "display": "inline-block",  # Ensure inline behavior
                "textAlign": "center"  # Center the "?" horizontally
            }
        ),

        # Tooltip for the question mark
        dbc.Tooltip(
            "This plot shows the top requesters for the selected year. Hover over the plot to explore details.",
            target="tooltip-target",
            placement="top",
        )
    ], style={'text-align': 'center'})  # Center the entire title block


# Callback to update the title for Plot X dynamically
@app.callback(
    Output('map_title', 'children'),
    Input('year-selector', 'value')
)
def update_map_title(selected_year):
    # Generate the dynamic title text
    title_text = f'Maplot of Year: {selected_year if selected_year != "all" else "All Time"}'

    # Return title with question mark and tooltip
    return html.Div([
        html.Span(title_text, style={
            'fontSize': '22px',
            'fontWeight': 'bold',
            'color': '#000000',
            'text-align': 'center'
        }),

        # The question mark icon with a tooltip
        html.Span(
            "?",
            id="tooltip-target",
            style={
                "cursor": "pointer",
                "color": "#ec0000",
                "fontWeight": "bold",
                "marginLeft": "10px",  # Space between title and question mark
                "fontSize": "20px",  # Make the question mark bigger
                "border": "2px solid #ec0000",  # Circular border
                "borderRadius": "50%",  # Make the border circular
                "padding": "0",  # Remove extra padding
                "width": "30px",  # Set fixed width
                "height": "30px",  # Set fixed height
                "lineHeight": "30px",  # Align text vertically
                "display": "inline-block",  # Ensure inline behavior
                "textAlign": "center"  # Center the "?" horizontally
            }
        ),

        # Tooltip for the question mark
        dbc.Tooltip(
            "This plot shows the top requesters for the selected year. Hover over the plot to explore details.",
            target="tooltip-target",
            placement="top",
        )
    ], style={'text-align': 'center'})  # Center the entire title block


@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('year-selector', 'value')]
)
def update_map(selected_year):
    df_filtered = grouped_df[grouped_df['Subsidiejaar'] == selected_year]
    df_filtered = df_filtered.set_index('Stadsdeel')
    df_filtered['geometry'] = df_filtered['WKT_LNG_LAT'].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df_filtered, geometry='geometry')
    geojson = json.loads(gdf.to_json())

    fig = px.choropleth_mapbox(gdf, geojson=geojson, locations=gdf.index, color=gdf['Bedragverleend'],
                               color_continuous_scale="tropic_r",
                               mapbox_style="carto-positron",
                               center={"lat": 52.357506, "lon": 4.928166}, opacity=0.3, zoom=10)

    fig.update_geos(fitbounds="locations")
    fig.update_layout(height=600)

    return fig


@app.callback(
    Output("prediction-output", "children"),
    [Input("predict-button", "n_clicks")],
    [dash.dependencies.State("amount-input", "value"),
     dash.dependencies.State("policy-dropdown", "value"),
     dash.dependencies.State("org-dropdown", "value"),
     dash.dependencies.State("periodicity-dropdown", "value")]
)

# Prediction function within the Dash framework
def predict_subsidy(n_clicks, amount, policy, org, periodicity):
    if n_clicks is None:
        return ""

    try:
        # Prepare input data
        input_data = pd.DataFrame([[amount, policy, org, periodicity]],
                                  columns=['Bedragaangevraagd', 'Beleidsterrein', 'Organisatieonderdeel',
                                           'Typeperiodiciteit'])

        # Make a prediction using the model
        probability = model.predict_proba(input_data)[0][1]
        probability_percentage = round(probability * 100, 2)

        # Display the prediction result
        return html.Div([
            html.H4(f"Prediction Result:"),
            html.P(f"There is a {probability_percentage}% chance of getting the subsidy.")
        ])

    except Exception as e:
        return html.Div(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

