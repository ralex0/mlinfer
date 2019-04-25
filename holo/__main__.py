"""
Holo. Program to display and interpret holograms through fitting and Bayesian
inference.

Usage:
  holo show <file> [--metadata <metadata>]
  holo fit <file> (--metadata <metadata>) [--method <method>] [-o <name>]
  holo infer <file> (--metadata <metadata>) [--method <method>] [-o <name>]
  holo report <file>
  holo -h | --help
  holo -v | --version

Options:
  -h --help     Show this screen.
  -v --version  Show version.
  --metadata    Specifiy metadata to display/fit image [default: '']
  --method      Specify fitting or inference method [default: '']
  -o <name>     Specify output file for results [default: '']
"""
import sys

from . import _argparser

def main():
    return get_args(cmd_args=sys.argv[1:])

def get_args(cmd_args=[]):
    arguments = _argparser.parse(__doc__, cmd_args=cmd_args)
    return arguments

if __name__ == '__main__':
    main()
