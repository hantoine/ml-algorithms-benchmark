from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError

def set_timeout(fct, timeout_time, use_signals=True, exception_message=None):
    return timeout(timeout_time, use_signals, exception_message=exception_message)(fct)