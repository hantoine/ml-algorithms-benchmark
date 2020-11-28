from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError


def set_timeout(fct, timeout_time):
    return timeout(timeout_time)(fct)
