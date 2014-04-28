def draw_pvalue(ax, pvalue, bars, yline, hline, vspace_text, digit_float=3):

    ytext = yline + vspace_text
    xtext = (max(bars) + float(min(bars))) / 2.

    ax.plot(bars, [yline, yline], linewidth=1.5, color='black')
    for i in bars:
        ax.plot([i, i], [yline, yline - hline], linewidth=1.5, color='black')
    ax.text(xtext, ytext, 'p < %.*f' % (digit_float, pvalue),
            horizontalalignment='center', verticalalignment='center',)


def axis_in_minute(ax):

    import matplotlib

    majorLocator = matplotlib.ticker.MultipleLocator(300)
    minorLocator = matplotlib.ticker.MultipleLocator(60)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)

    majorFormatter = matplotlib.ticker.FuncFormatter(
        lambda x, y: "%.0f" % (x / 60.))
    ax.xaxis.set_major_formatter(majorFormatter)
