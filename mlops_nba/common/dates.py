from datetime import date, datetime


def get_today():
    """Return today's date."""
    return date.today().strftime("%Y-%m-%d")


def get_now(for_files: bool = False):
    """Return today's date.

    Args:
        for_files (bool): if True, return now's date for files (without - between them)

    """
    now = datetime.now()
    if for_files:
        return now.strftime("%Y%m%d__%H%M%S")
    return now.strftime("%Y-%m-%d %H:%M:%S")
