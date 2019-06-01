from setuptools import setup

from mlinfer import __version__

setup(name='mlinfer',
      packages=['mlinfer'],
      version=__version__,
      entry_points={'console_scripts': ['mlinfer = mlinfer.__main__:main']})
