from importlib import reload

import multiframe
import matplotlib.pyplot as plt

reload(multiframe)

frm = multiframe.FrameMulti(3,2)

frm._params.show()
