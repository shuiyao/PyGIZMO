__all__ = ['is_installed_tqdm', 'import_tqdm']

_installed_tqdm = False

def static_vars(**kwargs):
    '''
    Decorate a function with static variables.

    Example:
        >>> @static_vars(counter=0)
        ... def foo():
        ...     foo.counter += 1
        ...     print('foo got called the %d. time' % foo.counter)
        >>> foo()
        foo got called the 1. time
        >>> foo()
        foo got called the 2. time
        >>> foo()
        foo got called the 3. time
    '''

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate

def is_installed_tqdm():
    '''Test whether h5py was properly imported by secure_get_h5py.'''
    global _installed_tqdm
    return _installed_tqdm

def _tqdm_fake_call(iterator, **kwargs):
    ''' 
    A faked tqdm call if tqdm is not installed.

    Returns
    -------
    iterator
    '''
    return iterator

@static_vars(_called=False)
def import_tqdm():
    '''
    Import tqdm if installed
    '''
    global _installed_tqdm
    try:
        from tqdm import tqdm
        _installed_tqdm = True
        return tqdm
    except:
        if not import_tqdm._called:
            import_tqdm._called = True
        return _tqdm_fake_call
