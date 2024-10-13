import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, dash_table
import plotly.express as px
import geopandas as gpd
from shapely import wkt
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('subsidie.csv')

# Remove rows where no subsidy is requested
df = df[df.Bedragaangevraagd != 0]
# replace all nan values of requested subsidy with 0
df = df.dropna(subset=['Bedragaangevraagd', 'Bedragverleend', 'Bedragvastgesteld'])

# Small df samlper
# df = df.sample(frac=0.1, replace=True, random_state=1)

# Filter rows where 'Organisatieonderdeel' contains 'stadsdeel' and doesn't contain 'Stadsdeeloverstijgend'
stadsdeel_data = df[df['Organisatieonderdeel'].str.contains('stadsdeel', case=False, na=False) &
                    ~df['Organisatieonderdeel'].str.contains('Stadsdeeloverstijgend', case=False, na=False)]

# Remove 'Stadsdeel ' from 'Organisatieonderdeel col values'
stadsdeel_data['Organisatieonderdeel'] = stadsdeel_data['Organisatieonderdeel'].str.replace('Stadsdeel ', '', case=False)

# Load geographic data for stadsdelen
geo_stadsdelen_df = pd.read_excel('stadsdelen.xlsx')

# Ensure the DataFrame only contains relevant stadsdelen (excluding 'Weesp' and 'Westpoort')
geo_stadsdelen_df = geo_stadsdelen_df[~geo_stadsdelen_df['Stadsdeel'].isin(['Weesp', 'Westpoort'])]

# Merge the two DataFrames on the 'Stadsdeel' column
merged_geo_df = pd.merge(geo_stadsdelen_df, stadsdeel_data, left_on='Stadsdeel', right_on='Organisatieonderdeel')

grouped_df = merged_geo_df.groupby(['Subsidiejaar', 'Stadsdeel']).agg({
    'Bedragverleend': 'sum',  # Sum the 'bedragverleend' column
    'WKT_LNG_LAT': 'first'  # Keep the first value of the 'wkt' column (assuming polygons are the same within groups)
}).reset_index()

# Function for total per year
df_sub_per_year = df.groupby("Subsidiejaar").sum()
float_cols = df_sub_per_year.select_dtypes(include=['float']).columns
df_sub_per_year[float_cols] = df_sub_per_year[float_cols].astype(int)
# calculate the percentage of the subisy request that is honoured
df_sub_per_year['Percentage_verleend'] = (df_sub_per_year['Bedragverleend'] / df_sub_per_year['Bedragaangevraagd'] * 100)

# Get unique years from the DataFrame
unique_years = sorted(df['Subsidiejaar'].unique())
# Create options for the RadioItems component
year_options = [{'label': str(year), 'value': year} for year in unique_years]
# Add an option for 'All Years'
year_options.insert(0, {'label': 'Alle Jaren', 'value': 'all'})



with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder_policy_area.pkl', 'rb') as file:
    label_encoder_policy_area = pickle.load(file)

with open('label_encoder_organizational_unit.pkl', 'rb') as file:
    label_encoder_organizational_unit = pickle.load(file)

with open('label_encoder_periodicity.pkl', 'rb') as file:
    label_encoder_periodicity = pickle.load(file)


def format_large_number(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} billion"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} million"
    elif number >= 1_000:
        return f"{number / 1_000:.2f} thousand"
    else:
        return str(number)

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Define the Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.Button("Dashboard", href="/", color="light")),  # Link to the dashboard ("/")
        dbc.NavItem(dbc.Button("Subsidy Predictor", href="/predict", color="light")),  # Link to the prediction page ("/predict")
        dbc.NavItem(dbc.Button("Page 3", href="/page3", color="light")),  # Link to Page 3 if needed
    ],
    brand="Amsterdam Subsidy Dashboard",
    brand_href="/",  # Clicking on the brand also links to the dashboard ("/")
    color="#ec0000",
    dark=True,
)
app.layout = html.Div([
    dcc.Location(id='url'),  # Keeps track of the URL
    navbar,
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/predict':
        return prediction_layout  # Prediction layout for "/predict"
    elif pathname == '/page3':
        return page_3_layout  # Some other layout for "/page3"
    else:
        return dashboard_layout  # Default to the dashboard layout ("/")




def create_card(p_text, h4_text):
    return dbc.Card([
        dbc.CardBody([
            html.P(p_text, className="card-value", style={
                'margin': '0px',
                'fontSize': '22px',
                'fontWeight': 'bold',
                'color': '#FFFFFF'
            }),
            html.H4(h4_text, className="card-title", style={
                'margin': '0px',
                'fontSize': '18px',
                'fontWeight': 'bold',
                'color': '#FFFFFF'
            })
        ], style={'textAlign': 'center'}),
    ], style={
        'paddingBlock': '10px',
        'backgroundColor': '#ec0000',
        'border': 'none',
        'borderRadius': '10px'
    })

card_year = dbc.Card([
    dbc.CardBody([
        # First, the current content (text elements)
        html.P("Select Year", className="card-value", style={
            'margin': '0px', 'fontSize': '20px', 'fontWeight': 'bold', 'color': '#FFFFFF'
        }),

        # Now, the Dropdown element
        dcc.Dropdown(
            id='year-selector',
            options=year_options,  # Use dynamic options from the DataFrame
            value='all',  # Default value
            clearable=False,  # Disallow clearing the selection
            style={'marginTop': '5px'}  # Add a bit of margin to separate it from the text
        )
    ], style={'textAlign': 'center'}),
], style={
    'paddingBlock': '2px',
    "backgroundColor": '#ec0000',
    'border': 'none',
    'borderRadius': '10px'
})

# Define the app layout with Navbar outside of the container
app.layout = html.Div([
    navbar,  # Navbar outside of the container to make it full width
    dbc.Container([
        dbc.Row([
            dbc.Col("card", width=6),
            dbc.Col(dbc.Stack([
                html.Div(card_year),
                html.Div(id='card-1-request'),
            ], gap=3), width=3),
            dbc.Col(dbc.Stack([
                html.Div(id='card-2-approval'),
                html.Div(id='card-3-grant'),
            ], gap=3), width=3),
            ], align="start", className="mb-4"
        ),
        dbc.Row([
            # dbc.Col(create_card("Pie chart", "Value"), width = 3),
            dbc.Col(dcc.Graph(id="pie_right"), width = 6),
            dbc.Col(html.Div(id="table"), width = 6),
            ], align="start", className="mb-4"
        ),
        dbc.Row([
            dbc.Col(dcc.Graph(id="choropleth-map"), width=8),
            dbc.Col(['Bar Chart'], width=4),
        ]),
    ], className="p-5")
])


# Callback to update the graph based on the selected year
# Define the callback to update multiple cards with the selected year
@app.callback(
    Output('card-1-request', 'children'),
    Output('card-2-approval', 'children'),
    Output('card-3-grant', 'children'),
    Input('year-selector', 'value')
)
def update_cards(selected_year):
    if selected_year == 'all':
        value_request = df_sub_per_year["Bedragaangevraagd"].sum()
        value_grant = df_sub_per_year["Bedragverleend"].sum()
        value_approval = round(((value_grant/value_request)*100), 2)
        return (create_card("Total Requested", str(format_large_number(value_request)) + " €"),
                create_card("Approval Rate", str(format_large_number(value_approval)) + "%"),
                create_card("Total Granted", str(format_large_number(value_grant)) + " €"))

    else:
        value_request = df_sub_per_year.loc[df_sub_per_year.index == selected_year, 'Bedragaangevraagd'].values[0]
        value_grant = df_sub_per_year.loc[df_sub_per_year.index == selected_year, 'Bedragverleend'].values[0]
        value_approval = round(df_sub_per_year.loc[df_sub_per_year.index == selected_year, 'Percentage_verleend'].values[0], 2)
        return (create_card("Total Requested", str(format_large_number(value_request)) + " €"),
                    create_card("Approval Rate", str(format_large_number(value_approval))+ "%"),
                    create_card("Total Granted", str(format_large_number(value_grant))+ " €"))

@app.callback(
    Output('table', 'children'),
    Input('year-selector', 'value')
)
def create_top_5_table(selected_year):
    # Filter the dataframe by year if a specific year is provided
    if selected_year != 'all':
        df_filtered = df[df['Subsidiejaar'] == selected_year]
    else:
        df_filtered = df

    # Group by 'Aanvrager' and 'Subsidiejaar', summing 'Bedragverleend' and counting applications ('Dossiernummer')
    df_grouped = df_filtered.groupby(['Aanvrager']).agg(
        total_bedragverleend=pd.NamedAgg(column='Bedragverleend', aggfunc='sum'),
        aanvraag_count=pd.NamedAgg(column='Dossiernummer', aggfunc='count')
    ).reset_index()

    # Sort by 'total_bedragverleend' and select the top 5
    df_top5 = df_grouped.sort_values(by='total_bedragverleend', ascending=False).head(6)

    # Apply function format large numbers
    df_top5['total_bedragverleend'] = df_top5['total_bedragverleend'].apply(format_large_number)
    # Create a Dash Bootstrap Table
    table = dbc.Table.from_dataframe(df_top5, striped=True, bordered=True, hover=True)

    return [html.Label(f'Top 7 Granted Requesters of {selected_year if selected_year != "all" else "All Time"}', style={
                'margin': '0px',
                'fontSize': '22px',
                'fontWeight': 'bold',
                'color': '#000000',
                'text-align': 'center',
                'display': 'block',
                'margin-bottom': '10px'
                    }),table]

@app.callback(
    Output('pie_right', 'figure'),
    Input('year-selector', 'value')
)

def plot_pie_chart_bedragverleend(selected_year):
    """Generates a pie chart showing the total 'Bedragverleend' per 'Beleidsterrein'."""
    # Filter the dataframe by year if a specific year is provided
    if selected_year != 'all':
        df_filtered = df[df['Subsidiejaar'] == selected_year]
    else:
        df_filtered = df

    # Group by 'Beleidsterrein' and sum 'Bedragverleend'
    df_grouped = df_filtered.groupby('Beleidsterrein').agg(
        total_bedragverleend=pd.NamedAgg(column='Bedragverleend', aggfunc='sum')
    ).reset_index()

    # Create a pie chart using Plotly
    fig = px.pie(df_grouped, values='total_bedragverleend', names='Beleidsterrein',
                 title=f'Total Bedragverleend per Beleidsterrein for Year {selected_year if selected_year != "all" else "All"}',
                 labels={'total_bedragverleend': 'Amount Granted'})

    # Show the plot
    return fig


# Callback to update the map based on selected year
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('year-selector', 'value')]
)
def update_map(Year_selected):
    # Filter the dataframe by the selected year
    df_filtered = grouped_df[grouped_df['Subsidiejaar'] == Year_selected]
    df_filtered = df_filtered.set_index('Stadsdeel')
    df_filtered['geometry'] = df_filtered['WKT_LNG_LAT'].apply(wkt.loads)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df_filtered, geometry='geometry')
    geojson = json.loads(gdf.to_json())

    # Create the Plotly choropleth map
    fig = px.choropleth_mapbox(gdf, geojson=geojson, locations=gdf.index, color=gdf['Bedragverleend'],
                               color_continuous_scale="tropic_r", title="Polygon Areas",
                               mapbox_style="carto-positron",
                               center={"lat": 52.357506, "lon": 4.928166}, opacity=0.3, zoom=10)

    fig.update_geos(fitbounds="locations")

    # Update the layout to increase the height
    fig.update_layout(height=600)

    return fig

# Dashboard layout
dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col("card", width=6),
        dbc.Col(dbc.Stack([
            html.Div(card_year),
            html.Div(id='card-1-request'),
        ], gap=3), width=3),
        dbc.Col(dbc.Stack([
            html.Div(id='card-2-approval'),
            html.Div(id='card-3-grant'),
        ], gap=3), width=3),
    ], align="start", className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="pie_right"), width=6),
        dbc.Col(html.Div(id="table"), width=6),
    ], align="start", className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="choropleth-map"), width=8),
        dbc.Col(['Bar Chart'], width=4),
    ]),
], className="p-5")



# Prepare dropdown options
policy_labels = label_encoder_policy_area.classes_
policy_options = [{'label': label, 'value': label} for label in policy_labels]

org_labels = label_encoder_organizational_unit.classes_
org_options = [{'label': label, 'value': label} for label in org_labels]

periodicity_labels = label_encoder_periodicity.classes_
periodicity_options = [{'label': label, 'value': label} for label in periodicity_labels]

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
                        dcc.Dropdown(id="policy-dropdown", options=policy_options),
                        width=8,
                    ),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Organizational Unit", html_for="org-dropdown", width=4),
                    dbc.Col(
                        dcc.Dropdown(id="org-dropdown", options=org_options),
                        width=8,
                    ),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Periodicity Type", html_for="periodicity-dropdown", width=4),
                    dbc.Col(
                        dcc.Dropdown(id="periodicity-dropdown", options=periodicity_options),
                        width=8,
                    ),
                ], className="mb-3"),
                dbc.Button("Predict", color="primary", id="predict-button", className="mt-3"),
            ]),
        ], md=6),
        dbc.Col([
            html.Div(id="prediction-output", className="mt-4"),
            html.Div(id="recommendation-output", className="mt-4")
        ], md=6),
    ]),
])

# Precompute statistics for recommendations
df_recommend = pd.read_csv('subsidies_openbaar_subsidieregister.csv')
df_recommend = df_recommend.dropna(subset=['Bedragaangevraagd', 'Bedragverleend', 'Beleidsterrein', 'Organisatieonderdeel', 'Typeperiodiciteit'])
df_recommend['Approved'] = (df_recommend['Bedragverleend'] > 0).astype(int)

# Encode categorical variables
df_recommend['Beleidsterrein'] = label_encoder_policy_area.transform(df_recommend['Beleidsterrein'])
df_recommend['Organisatieonderdeel'] = label_encoder_organizational_unit.transform(df_recommend['Organisatieonderdeel'])
df_recommend['Typeperiodiciteit'] = label_encoder_periodicity.transform(df_recommend['Typeperiodiciteit'])

# Precompute approval rates for recommendations
mean_amount_approved = df_recommend[df_recommend['Approved'] == 1]['Bedragaangevraagd'].mean()

approval_rates_by_policy = df_recommend.groupby('Beleidsterrein')['Approved'].mean()
top_policies = approval_rates_by_policy.sort_values(ascending=False).head(3).index.tolist()

approval_rates_by_org = df_recommend.groupby('Organisatieonderdeel')['Approved'].mean()
top_orgs = approval_rates_by_org.sort_values(ascending=False).head(3).index.tolist()

approval_rates_by_periodicity = df_recommend.groupby('Typeperiodiciteit')['Approved'].mean()
top_periodicities = approval_rates_by_periodicity.sort_values(ascending=False).head(1).index.tolist()


@app.callback(
    [Output("prediction-output", "children"),
     Output("recommendation-output", "children")],
    [Input("predict-button", "n_clicks")],
    [dash.dependencies.State("amount-input", "value"),
     dash.dependencies.State("policy-dropdown", "value"),
     dash.dependencies.State("org-dropdown", "value"),
     dash.dependencies.State("periodicity-dropdown", "value")]
)
def predict_subsidy_with_recommendations(n_clicks, amount, policy, org, periodicity):
    if n_clicks is None:
        return "", ""

    try:
        # Check for missing inputs
        if None in (amount, policy, org, periodicity):
            return html.Div("Please fill in all input fields."), ""

        # Encode categorical variables
        policy_encoded = label_encoder_policy_area.transform([policy])[0]
        org_encoded = label_encoder_organizational_unit.transform([org])[0]
        periodicity_encoded = label_encoder_periodicity.transform([periodicity])[0]

        # Prepare input data
        input_data = pd.DataFrame([[amount, policy_encoded, org_encoded, periodicity_encoded]],
                                  columns=['Bedragaangevraagd', 'Beleidsterrein', 'Organisatieonderdeel',
                                           'Typeperiodiciteit'])

        # Make a prediction using the model
        probability = model.predict_proba(input_data)[0][1]
        probability_percentage = round(probability * 100, 2)

        # Prediction output
        prediction_result = html.Div([
            html.H4(f"Prediction Result:"),
            html.P(f"There is a {probability_percentage}% chance of getting the subsidy.")
        ])

        # Recommendations (simplified logic here)
        recommendations = []
        if amount > mean_amount_approved:
            recommendations.append(f"Consider reducing the requested amount to improve approval chances.")
        if policy_encoded not in top_policies:
            recommendations.append("Consider aligning with higher approval policy areas.")

        recommendation_output = html.Div([
            html.H4("Recommendations:"),
            html.Ul([html.Li(rec) for rec in recommendations])
        ])

        return prediction_result, recommendation_output

    except Exception as e:
        return html.Div(f"An error occurred: {str(e)}"), ""


if __name__ == '__main__':
    app.run(debug=True)