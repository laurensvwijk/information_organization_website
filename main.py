import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, dash_table
import plotly.express as px
import geopandas as gpd
from shapely import wkt
import json

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
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Define the Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.Button("Page 1", href="#", color="light")),
        dbc.NavItem(dbc.NavLink("Page 2", href="#")),
        dbc.NavItem(dbc.NavLink("Page 3", href="#")),
    ],
    brand="Amsterdam Subsidy Dashboard",
    brand_href="#",
    color="#ec0000",
    dark=True,
)


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


if __name__ == '__main__':
    app.run(debug=True)