import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

# make figure and assign axis objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
fig.subplots_adjust(wspace=0)

# pie chart parameters
overall_ratios = [1, 1, 1]
labels = ["Quality", "Scale", "Movement"]
explode = [0.1, 0, 0]
# rotate so that first wedge is split by the x-axis
angle = -60 * overall_ratios[0]
wedges, *_ = ax1.pie(
    overall_ratios,
    startangle=angle,
    labels=labels,
    explode=explode,
    textprops={"fontsize": 14},
)

# bar chart parameters
ratios = [0.33, 0.33, 0.33, 0.01]
labels = ["Exposure", "Contrast", "Focus", "Other"]
bottom = 1
width = 0.2

# Adding from the top matches the legend.
for j, (height, label) in enumerate(reversed([*zip(ratios, labels)])):
    bottom -= height
    bc = ax2.bar(
        0, height, width, bottom=bottom, color="C0", label=label, alpha=0.1 + 0.25 * j
    )

ax2.set_title("Metrics of Quality")
ax2.legend(fontsize=12, loc="upper right", bbox_to_anchor=(1.1, 1))
ax2.axis("off")
ax2.set_xlim(-2.5 * width, 2.5 * width)

# use ConnectionPatch to draw lines between the two plots
theta1, theta2 = wedges[0].theta1, wedges[0].theta2
center, r = wedges[0].center, wedges[0].r
bar_height = sum(ratios)

# draw top connecting line
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(
    xyA=(-width / 2, bar_height),
    coordsA=ax2.transData,
    xyB=(x, y),
    coordsB=ax1.transData,
)
con.set_color([0, 0, 0])
con.set_linewidth(2)
ax2.add_artist(con)

# draw bottom connecting line
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(
    xyA=(-width / 2, 0), coordsA=ax2.transData, xyB=(x, y), coordsB=ax1.transData
)
con.set_color([0, 0, 0])
ax2.add_artist(con)
con.set_linewidth(2)

plt.show()
