from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError

def set_timeout(fct, timeout_time):
    @timeout(timeout_time)
    def fct_with_timeout(*args, **kargs):
        return fct(*args, **kargs)
    return fct_with_timeout