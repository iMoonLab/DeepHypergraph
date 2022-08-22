from math import log, exp


def C(n, m):
    r""" Compute the C(n, m) function.
        log(C(n, m)) = log(n!/(n-m)!) - log m!
                     = log(n-m+1) + log(n-m+2) + ...
                        - (log 1 + log 2 + ...)
    """
    log_value = 0
    m = min(m, n - m)
    for i in range(1, m + 1):
        log_value = log_value + log(n - m + i) - log(i)
    return int(round(exp(log_value), 0))
