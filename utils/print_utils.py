import datetime as dt


def __try_log(msg, severity):
    time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    str_out = '|%s| [%s]\t%s' % (severity, time_str, msg)
    print(str_out)

def try_log_trace(msg):
    __try_log(msg, 'TRACE')

def try_log_debug(msg):
    __try_log(msg, 'DEBUG')

def try_log_info(msg):
    __try_log(msg, 'INFO')

def try_log_warn(msg):
    __try_log(msg, 'WARN')

def try_log_error(msg):
    __try_log(msg, 'ERROR')

def try_log_severe(msg):
    __try_log(msg, 'SEVERE')