"""
Holo. Program to display and interpret holograms through fitting and Bayesian
inference.

Usage:
  holo show <file> ... [-m | --metadata] <metadata>
  holo fit <file> (-m | --metadata) <metadata> [--method <method>] [-o <outfile>]
  holo infer <file> (-m | --metadata) <metadata> [--method <method>] [-o <outfile>]
  holo report <file> ... [-m | --metadata] <metadata>
  holo (-h | --help)
  holo (-v | --version)

Options:
  -h --help     Show this screen.
  -v --version  Show version.
  --metadata    Specifiy metadata to display/fit image [default: '']
  --method      Specify fitting or inference method [default: '']
  -o <outfile>  Specify output file for results [default: '']
"""
from . import _argparser
from . import __version__

def main():
    arguments = _argparser.parse(__doc__, version='Holo {}'.format(__version__))
    print(arguments)

if __name__ == '__main__':
    main()
