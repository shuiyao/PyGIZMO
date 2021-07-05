# from importlib import reload

# import multiframe
# import matplotlib.pyplot as plt

# reload(multiframe)

# frm = multiframe.FrameMulti(3,2)

# frm._params.show()

import abc

class Base(abc.ABC):
    def __init__(self):
        print("Base.__init__()")
        self._supername = "SUPER"

    @abc.abstractmethod
    def name(self):
        pass

    def supername(self):
        print(self._supername)

    def myname(self):
        print(self._supername)        

class Derived(Base):
    def __init__(self, name):
        super(Derived, self).__init__()
        print("Derived.__init__()")
        self._name = name

    # implementation
    def name(self):
        print(self._name)

    # overload
    def myname(self):
        print(self._name)

# base = Base()
# Can't instantiate abstract class Base with abstract method name
derived = Derived("Derived Class")

derived.name()

derived.supername()

derived.myname()
