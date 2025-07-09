
from shiny import App, ui, render, reactive
from plotly.callbacks import Points
from shinywidgets import output_widget, render_plotly, render_widget

import random
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import defaultdict 
from sklearn.manifold import MDS
import os

# bubble plot data
# df = pd.read_csv(Path("area2category_score_iui.csv"), index_col=["campus", "area_shortname", "area"])
df = pd.read_csv(Path("area2category_score_campus.csv"), index_col=["campus", "area_shortname", "area"])
df_norm = df.div(df.sum(axis=1), axis=0)
df['category'] = df.idxmax(axis=1)
df['size'] = 80 # bubble size
df = df.reset_index()
df['area_campus'] = df['area_shortname'] + "<br>(" + df['campus'] + ")"

# embed score w/ MDS
mds_seed = 2971 #4115, 8387, 2971
print(f"mds seed: {mds_seed}")
# mds_seed = random.randint(0, 10000)
# print(f"mds random seed: {mds_seed}")
embedding = MDS(n_components=2, n_init=4, random_state=mds_seed).fit_transform(df_norm)
embedding_df = pd.DataFrame(embedding, columns=["x", "y"])
embedding_df = (embedding_df-embedding_df.min())/(embedding_df.max()-embedding_df.min())
embedding_df = pd.concat([embedding_df, df[["area_campus", "campus", 'area', 'area_shortname', 'category', 'size']]], axis=1)

# extra info: area pis & links
area2pi2url = pd.read_csv(Path("area2pi2url.csv"))
area2pis_dict = defaultdict(list)
for area in area2pi2url['area'].unique():
  area2pis_dict[area] = area2pi2url[area2pi2url['area'] == area]['pi'].tolist()
pi2url_df = area2pi2url[['pi', 'url']].drop_duplicates()
pi2url_dict = pi2url_df.set_index('pi')['url'].to_dict()


app_ui = ui.page_fluid(
    ui.tags.style("""
        .main-row { 
          display: flex; 
          flex-direction: row; 
          height: 100vw; 
          width: 100vw; 
        }
        .sidebar {
          flex: 1 1 30%;
          min-width: 250px; 
          max-width: 400px; 
          # background: #f9f9f9; 
          border-left: 1px solid #eee; 
          padding: 24px; overflow-y: auto; 
        }
        .plot-area { 
          flex: 3 1 70%; 
          aspect-ratio: 4/3; 
          width: 70vw; 
          min-width: 60vw; 
          height: 100%; 
          position: relative;
        }
        .plotly-graph-div { 
          height: 100% !important; 
          width: 100% !important; 
        }
        #bubble {
        height: 60vw !important;
        width: 70vw !important;
        max-height: 100vh !important;
        }
        /* Disable pointer cursor on legend items */
        .legend .traces .legendtoggle {
          cursor: default !important;
        }
        /* Alternative selector that might be needed */
        g.legend g.traces g.legendtoggle {
          cursor: default !important;
        }
        /* More specific selector for plotly legend */
        .plotly .legend .traces .legendtoggle {
          cursor: default !important;
        }
    """),
    ui.div(
        ui.div(output_widget("bubble"), class_="plot-area"),
        ui.div(
          ui.output_ui("click_info"),
          class_="sidebar"),
        class_="main-row"
    )
)

def server(input, output, session):
  click_reactive = reactive.value()
  
  @render_widget
  def bubble():
    import plotly.graph_objects as go
    # colors = ["blue", "green", "red", "orange", "purple", "gray", "brown", ]
    colors = sns.color_palette("tab10", n_colors=10).as_hex()
    cat2color_dict = dict(zip(embedding_df["category"].unique(), colors))
    categories = sorted(embedding_df["category"].unique())
    embedding_df["category_color"] = embedding_df["category"].map(cat2color_dict)
    hover_text = embedding_df["area"]

    fig = go.FigureWidget(
      go.Scatter(
        x=embedding_df["x"],
        y=embedding_df["y"],
        mode="markers+text",
        marker=dict(
            size=embedding_df["size"],
            sizemode='area',
            sizeref=2.*max(embedding_df["size"])/(100.**2),  # scale size_max=60
            opacity=0.1,
            color=embedding_df["category_color"],
        ),
        text=embedding_df["area_campus"],
        hovertext=hover_text,
        hoverinfo="text",  # Only show hovertext
        showlegend=False,
        customdata=embedding_df[["area", "category"]].values,
      )
    ) #FigureWidget

    # Add invisible traces for legend entries
    for cat in categories:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],  # No actual data points
                mode="markers",
                marker=dict(color=cat2color_dict[cat], 
                            opacity=0.4,
                            size=10),
                name=cat,
                showlegend=True,
            )
        )
    fig.update_layout(
        autosize=True,
        # width=800, height=800, ## comment out for auto height
        # width=None, height=None, ## comment out for auto height
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(255,255,255,0.)",
            font=dict(color="darkslategray"),
        ),
        title=dict(
            text="click bubble to see PIs",
            font={"size": 12, "color": "darkslategray"},
            x=0, xanchor="left",
            y=0.98, yanchor="top",
        ), 
        legend=dict(
          orientation="h",
          x=1, xanchor="right", xref="paper",
          y=0.96, yanchor="top", yref="container",
          bgcolor="rgba(0,0,0,0)", # Transparent background
          entrywidthmode='fraction', entrywidth=.2,
          itemclick=False, itemdoubleclick=False, # disable legend interactivity
          font=dict(color="darkslategray"),
        ),
        legend_title_text="",
    )

    fig.data[0].on_click(on_point_click)
    return fig
    
  def on_point_click(trace, points, state): 
    idx = points.point_inds[0]
    # Use customdata to get the area information, which is more reliable
    # since we stored [area, category] in customdata for each point
    area = trace.customdata[idx][0]  # area is the first element in customdata
    pis = area2pis_dict.get(area, [])
    pis = sorted(pis)
    # Each PI line: <a href="URL" target="_blank" title="URL">PI NAME</a><br>
    pi_url_str = "".join([f'<a href="{pi2url_dict[pi]}" target="_blank" title={pi2url_dict[pi]}>{pi}</a><br>' for pi in pis])
    info = ""
    if area not in ["The Polis Center", 
                    "Integrated Nanosystems Development Institute (INDI)"]:
      info += f"<br>{area}"

    info += f"<br>{pi_url_str}"
    click_reactive.set(info)

  @render.text
  def click_info():
    return click_reactive.get()

    

app = App(app_ui, server)

