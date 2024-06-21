import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("dark_background")
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False

dark2 = sns.color_palette('Dark2')


def plot_cat_count(sr_data, x_label, y_label, title, index_on_x=1, xticklbl_rotate=0):
    fig = plt.figure(figsize=(8, 5))
    if index_on_x == 1:
        ax = sns.barplot(x=sr_data.index, y=sr_data.values, hue=sr_data.values, legend=False, palette='Dark2')
    else:
        ax = sns.barplot(x=sr_data.values, y=sr_data.index, hue=sr_data.values, legend=False, palette='Dark2')
    ax.set_xlabel(x_label, fontdict={'fontsize': 10})
    ax.set_ylabel(y_label, fontdict={'fontsize': 10})
    ax.set_title(title, fontdict={'fontsize': 10, 'fontweight': 'bold'})
    ax.tick_params(labelsize=10)
    if xticklbl_rotate > 0:
        ax.set_xticks(sr_data.index)
        ax.set_xticklabels(labels=sr_data.index, rotation=xticklbl_rotate)
    plt.tight_layout()
    plt.show()
