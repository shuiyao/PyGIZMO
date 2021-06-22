__all__ = ['ParamsTopLevel', 'ParamsMulti']
import abc

class ParamsTopLevel(abc.ABC): # default set of parameters, the object defines it as a new-type
    def __init__(self):
        self.num = 1
        self.figsize = (6.4, 4.8)
        self.dpi = 100
        self.top = 0.88
        self.bottom = 0.11
        self.right = 0.9
        self.left = 0.125
        self.hspace = 0.2
        self.wspace = 0.2
        self.fontsize_label = 12
        self.fontsize_tick = 8
        self.fontsize_legend = 10
        self.ylabels = ""
        self.xlabels = ""
        self.suptitle = ""

    @staticmethod
    def pixels_to_inches(pix, dpi):
        '''
        Convert pixels to inches.
        dpi: pixel per inch
        '''
        return pix / dpi

    def _pixels_fignorm(self, pix, axis=0):
        '''
        Convert pixels to dimensional unit relative to a figure.
        '''
        if(axis == 'x'): axis = 0
        elif(axis == 'y'): axis = 1
        
        norm = self.figsize[axis]
        return ParamsTopLevel.pixels_to_inches(pix, self.dpi) / norm

    @staticmethod
    def _print_param(key, val):
        print("{:16s}: {}".format(key, val))

    @abc.abstractmethod
    def show(self):
        print ("Parameters")
        print ("--------------------------------")
        self._print_param("figsize", self.figsize)
        self._print_param("dpi", self.dpi)
        self._print_param("top", self.top)
        self._print_param("bottom", self.bottom)
        self._print_param("right", self.right)
        self._print_param("left", self.left)
        self._print_param("hspace", self.hspace)
        self._print_param("wspace", self.wspace)
        self._print_param("fontsize_label", self.fontsize_label)
        self._print_param("fontsize_tick", self.fontsize_tick)
        self._print_param("fontsize_legend", self.fontsize_legend)        
        self._print_param("ylabels", self.ylabels)
        self._print_param("xlabels", self.xlabels)
        self._print_param("suptitle", self.suptitle)

    def help(self):
        self.show()

class ParamsMulti(ParamsTopLevel):
    def __init__(self, nrows, ncols, tight_layout=True):
        super(ParamsMulti, self).__init__() # Have to define ParamsTopLevel as ParamsTopLevel(object)
        self.left = 0.12
        self.dims = (nrows, ncols)
        self.npanels = nrows * ncols
        self.height_ratios = [1] * nrows
        self.width_ratios = [1] * ncols    
        self.figsize = (4 * ncols, 3 * nrows)
        if(tight_layout):
            self.hspace = 0.0
            self.wspace = 0.0
        else:
            self.right = 0.95
            self.bottom = 0.10
            self.top = 0.92
            self.hspace = 0.20
            self.wspace = 0.25

    def show(self):
        super().show()
        self._print_param("dims", self.dims)                
        self._print_param("npanels", self.npanels)
        self._print_param("height_ratios", self.height_ratios)
        self._print_param("width_ratios", self.width_ratios)
        
par = ParamsMulti(2, 2)
