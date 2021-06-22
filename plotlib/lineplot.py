import configparser

class LinePlot():
    '''
    BaseClass for line-type plot, such as the galactic stellar mass functions
    (GSMFs), mass-metallicity relations (MZRs), etc.
    '''
    def __init__(self, ax):
        pass

    def load_config(self, plottype='GSMF', path_config=None):
        if(path_config is None):
            print("load default configuration file.")
            path_config = "lineplot.cfg"
        cfg = configparser.ConfigParser()
        cfg.optionxform=str
        cfg.read(path_config)

        try:
            pars = cfg[plottype]
        except:
            raise KeyError("plottype ({}) not in {}".format(plottype, list(cfg.keys())))

        self.xlims = (float(pars.get('xmin')), float(pars.get('xmax')))
        self.ylims = (float(pars.get('ymin')), float(pars.get('ymax')))
        self.xlim_lres = pars.get('xlim_lres')
        if(self.xlim_lres is not None): self.xlim_lres = float(self.xlim_lres)
        self.xlim_hres = pars.get('xlim_hres')
        if(self.xlim_hres is not None): self.xlim_hres = float(self.xlim_hres)
        self.xlabel = pars.get('xlabel')
        self.ylabel = pars.get('ylabel')
        self.legend_model = pars.get('legend_model')
        self.legend_data = pars.get('legend_data')
        self.xticks = [float(x) for x in pars.get('xticks').split(sep=',')]
        self.yticks = [float(y) for y in pars.get('yticks').split(sep=',')]        

    def add_legend_model(loc='upper right'):
        '''
        Add the legends of different models to the plot.
        '''
        self.legend_model.draw(loc=loc)

    def add_legend_data(loc='lower right'):
        '''
        Add the legends of different observational data to the plot.
        '''
        self.legend_data.draw(loc=loc)

    def add_colorbar(loc='bottom'):
        pass

    def add_data(self, data, errorbar=True, add_legend=True):
        # Should be different for different kinds of plot
        x, y = self.load_data()

    def add_model(self, model, errorbar=True, add_legend=True):
        # should be different for different plot
        x, y = self.load_model_data(self.desc[model]['path'])
        name = self.desc[model]['name']
        lc = self.desc[model]['linecolor']
        lw = self.desc[model]['linewidth']
        ls = self.desc[model]['linestyle']
        marker = self.desc[model]['marker']
        ms = self.desc[model]['markersize']
        if(ms > 0):
            ax.plot(x, y, marker=marker, color=lc, markersize=ms)
        ax.plot(x, y, color=lc, linestyle=linestyle, linewidth=linewidth)
        ax.errorbar(x, y)

        self.model.append(model)
        if(add_legend):
            self.legend_model.add_line(name, lc, lw, ls)

ax=None
line = LinePlot(ax)
line.load_config()
