from setuptools import setup

from holo import __version__

setup(name='holo',
      version=__version__,
      packages=['holo'],
      entry_points={'console_scripts': ['holo = holo.__main__:main']})
