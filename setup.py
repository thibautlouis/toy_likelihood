from setuptools import setup, find_packages
from pslike import __author__, __version__, __url__

setup(name="toy_likelihood",
      version = __version__,
      packages = find_packages(),
      description = "Preliminary SO likelihood",
      url = __url__,
      author = __author__,
      keywords = ["CMB", "SO"],
      classifiers = ["Intended Audience :: Science/Research",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Operating System :: OS Independent",
                     "Programming Language :: Python :: 2.7",
                     "Programming Language :: Python :: 3.7"],
      install_requires = ["camb", "cobaya"],
      entry_points = {
        "console_scripts": ["pslike=pslike.pslike:main"],
      }
)
