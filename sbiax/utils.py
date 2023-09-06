from chainconsumer import ChainConsumer


def make_plot(samples, labels, params_name, truth):
    c = ChainConsumer()

    for i, data in enumerate(samples):
        c.add_chain(
            data,
            parameters=params_name,
            name=labels[i],
            shade_alpha=0.5
        )

    c.configure(legend_kwargs={"fontsize": 20}, tick_font_size=8, label_font_size=20)

    fig = c.plotter.plot(figsize=0.8, truth=truth)
