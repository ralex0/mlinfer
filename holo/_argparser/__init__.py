"""
Argument parser for entry scripts

# TODO : Decide if should refactor to use standard library argparse instead of 
         docopts
"""
from . import docopt as docopt
from .. import __version__

def parse(doc, cmd_args):
    argv = ['-h'] if len(cmd_args) == 0 else cmd_args
    pargs = docopt.docopt(doc, argv=argv, version='Holo {}'.format(__version__))
    return pargs
