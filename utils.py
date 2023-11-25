import os


def make_dir(timestamp):
    """Make directory if it does not exist."""
    if not os.path.exists(timestamp):
        os.makedirs(timestamp)