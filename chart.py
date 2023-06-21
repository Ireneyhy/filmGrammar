import matplotlib.pyplot as plt
import seaborn as sns


colors = sns.color_palette("Set2")

labels = "Movement", "Scale", "Quality"
sizes = [1, 1, 1]

fig, ax = plt.subplots()
ax.pie(
    sizes,
    labels=labels,
    startangle=90,
    colors=colors,
    labeldistance=0.4,
    textprops={"fontsize": 13},
)
ax.axis("equal")
plt.show()
