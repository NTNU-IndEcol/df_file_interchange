"""
Common functions that don't fit anywhere else.
"""



def str_n(in_str):
    """Does a simple str cast but if None converts to empty str

    Parameters
    ----------
    in_str : Any

    Returns
    -------
    str
        str(in_str) or "" if in_str == None
    """

    if in_str is None:
        return ""
    else:
        return str(in_str)
    


def safe_str_output(in_qty, truncate_len: int = 200):
    """Allows dumping of variables to log files, etc, in a vaguely sane/safe way

    Tries conversion to str and then truncate to max `truncate_len` chars. If
    that fails, returns empty str.

    Parameters
    ----------
    in_qty : _type_
        _description_

    Returns
    -------
    str
        Hopefully, a meaningful and safe str representation of the variable.
    """

    secured_output = ""
    try:
        secured_output = str(in_qty)
        if len(secured_output) > truncate_len:
            secured_output = secured_output[0:truncate_len]
    except:
        pass

    return safe_str_output