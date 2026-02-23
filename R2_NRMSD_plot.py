

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FILE_PATH = r'I:/My Drive/Personal Webpage/Github/MCDM_Analysis/Basin_Averaged_Performance_Indicators_From_BiasCorrected.csv'
df = pd.read_csv(FILE_PATH)

# variables: use CC -> R2, NRMSD -> RMSD metric
vars_map = {
    "Pr":   {"cc": "CC_Pr",     "rmsd": "NRMSD_Pr"},
    "Tmax": {"cc": "CC_Tasmax", "rmsd": "NRMSD_Tasmax"},
    "Tmin": {"cc": "CC_Tasmin", "rmsd": "NRMSD_Tasmin"},
}

var_colors = {"Pr":"tab:blue", "Tmax":"tab:orange", "Tmin":"tab:green"}

models = df["Model"].astype(str).tolist()

# marker per model
marker_pool = ["o","s","^","v","P","X","*","<",">","h","H","d","p","8","1","2","3","4","|","_"]
marker_map = {m: marker_pool[i % len(marker_pool)] for i, m in enumerate(models)}

fig, ax = plt.subplots(figsize=(9, 6))

for vname, cols in vars_map.items():
    r = df[cols["cc"]].astype(float).values
    r2 = np.clip(r, -1, 1)**2
    rmsd = df[cols["rmsd"]].astype(float).values  # your NRMSD

    for i, m in enumerate(models):
        ax.plot(
    rmsd[i],
    r2[i],
    marker=marker_map[m],
    linestyle="None",

    markerfacecolor='none',                 # makes marker hollow
    markeredgecolor=var_colors[vname],      # colored outline
    markeredgewidth=1.6,                    # thickness of outline

    markersize=8,
    alpha=1
)

ax.set_xlabel("NRMSD (lower is better)")
ax.set_ylabel("R² (higher is better)")
ax.set_title("Model Skill Plot: R² vs NRMSD (Color=Variable, Symbol=Model)")
ax.grid(True)

# legends
var_handles = [
    Line2D([0],[0], marker="o", linestyle="None",
           markerfacecolor='none',
           markeredgecolor=var_colors["Pr"],
           markeredgewidth=1.4,
           label="Pr", markersize=9),

    Line2D([0],[0], marker="o", linestyle="None",
           markerfacecolor='none',
           markeredgecolor=var_colors["Tmax"],
           markeredgewidth=1.4,
           label="Tmax", markersize=9),

    Line2D([0],[0], marker="o", linestyle="None",
           markerfacecolor='none',
           markeredgecolor=var_colors["Tmin"],
           markeredgewidth=1.4,
           label="Tmin", markersize=9),
]

model_handles = [
    Line2D([0],[0],
           marker=marker_map[m],
           linestyle="None",
           markerfacecolor='none',
           markeredgecolor="black",
           markeredgewidth=1.2,
           label=m,
           markersize=9)
    for m in models
]
leg1 = ax.legend(handles=var_handles, loc="upper right", title="Variable")
ax.add_artist(leg1)
ax.legend(handles=model_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Model")

plt.tight_layout()
plt.show()

#%%

