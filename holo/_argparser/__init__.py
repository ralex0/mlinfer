"""
Argument parser for entry script

# TODO : Decide if should refactor to use standard library argparse instead of 
         docopts
"""
from . import docopt as docopt

def parse(*args, **kwargs):
    return _parse(*args, **kwargs)
    
def _parse(*args, **kwargs):
    return docopt.docopt(*args, **kwargs)
