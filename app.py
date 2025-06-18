
from shiny import App, ui, render, reactive
from plotly.callbacks import Points
from shinywidgets import output_widget, render_widget, render_plotly

import random
import pandas as pd
from pathlib import Path
from collections import defaultdict 
from sklearn.manifold import MDS
import os

pi2area_df = pd.read_csv(Path("pi2area.csv"))
pi2area_dict = {}
for area in pi2area_df.area.unique():
  pi2area_dict[area] = pi2area_df[pi2area_df.area == area].pi.unique().tolist()

pi2url_df = pd.read_csv(Path("pi2url.csv"))
pi2url_dict = dict(zip(pi2url_df.pi, pi2url_df.url))

proj2area_df = pd.read_csv(Path("proj2area.csv"))
area2cat_df = pd.read_csv(Path("area2category.csv"))
area2cat = dict(zip(area2cat_df['area'], area2cat_df['category']))
area2cat_df['category'] = area2cat_df['area'].map(lambda x: area2cat[x])
area2cat_df['size'] = 60

proj2area = defaultdict(list)
for _, row in proj2area_df.iterrows():
  proj2area[row["project"]].append(row["area"])

df = pd.read_csv(Path("area2category_count_fang.csv"), index_col="area")
mds_seed = random.randint(0, 10000)
print(f"mds random seed: {mds_seed}")
df_norm = df.div(df.sum(axis=1), axis=0)
embedding = MDS(n_components=2, n_init=4, random_state=mds_seed).fit_transform(df_norm)
embedding_df = pd.DataFrame(embedding, columns=["x", "y"])
embedding_df = (embedding_df-embedding_df.min())/(embedding_df.max()-embedding_df.min())
embedding_df = pd.concat([embedding_df, area2cat_df[['area', 'area_short', 'category', 'size']]], axis=1)

# app_ui = ui.page_fluid(
#   # triple quotes define multi-line string literals, allowing multi-lines CSS
#   ui.tags.style(""" 
#       .plotly-graph-div {
#         height: 100% !important;
#         width: 100% !important;
#       }
#       #bubble {
#         height: 90vw !important;
#         width: 100vw !important;
#       }
#     """), 
#     ui.div(
#       output_widget("bubble"),
#       id="bubble",
#       # style="height: 100vh; width: 100vw;"
#     ),
#   "click bubble to see PIs",
#   ui.output_ui("click_info")
# )

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
    """),
    # ui.div("click bubble to see PIs", style="font-size: 1.5em;"), 
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
  
  @render_plotly
  # @render_widget
  def bubble():
    # import plotly.express as px
    import plotly.graph_objects as go
    
    colors = ["blue", "green", "red", "orange", "purple", "gray", "brown"]
    marker_colors = embedding_df["category"].map(dict(zip(embedding_df["category"].unique(), colors)))
    hover_text = embedding_df["area"]
    
    fig = go.FigureWidget( #FigureWidget
      go.Scatter(
        x=embedding_df["x"],
        y=embedding_df["y"],
        mode="markers+text",
        marker=dict(
            size=embedding_df["size"],
            sizemode='area',
            sizeref=2.*max(embedding_df["size"])/(100.**2),  # scale size_max=60
            opacity=0.05,
            color=marker_colors
        ),
        text=embedding_df["area_short"],
        # textposition="top center",   # or "none" to disable
        hovertext=hover_text,
        hoverinfo="text",  # Only show hovertext
        showlegend=False,
        customdata=embedding_df[["area", "category"]].values,
      )
    )

    fig.update_layout(

        autosize=True,
        # width=800, height=800, ## comment out for auto height
        # width=None, height=None, ## comment out for auto height
        # showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(255,255,255,0.)",
            font=dict(color="darkslategray"),
        ),
        title={
            "text": "click bubble to see PIs",
            "font": {"size": 12, "color": "darkslategray"},
            "x": 0,  # Center the title
            "xanchor": "left"
        }, 
                margin=dict(l=0, r=0, t=50, b=0),
    )
    
    fig.data[0].on_click(on_point_click)
    
    return fig
    
  def on_point_click(trace, points, state): 
    idx = points.point_inds[0]
    area = embedding_df.iloc[idx]["area"]
    pis = pi2area_dict.get(area, [])
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
