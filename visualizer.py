import pandas as pd
import plotly.graph_objects as go

data = []
for p in particle_list:
  x = [v[0] for v in p.position_history]
  y = [v[1] for v in p.position_history]
  z = [v[2] for v in p.position_history]
  data.append(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers', marker=dict(color='white', size=3)))
  data.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='yellow', width=2.5)))


fig = go.Figure(data)

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    paper_bgcolor='black',
    plot_bgcolor='black',
    scene=dict(
        xaxis=dict(showbackground=False, visible=False),
        yaxis=dict(showbackground=False, visible=False),
        zaxis=dict(showbackground=False, visible=False),
    ),
    font=dict(color='white')
)

fig.show()
