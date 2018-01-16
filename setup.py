from setuptools import setup

setup(name='swissknife',
      version='0.1',
      description='Various tools to handle data and simmulations',
      url='http://github.com/zekearneodo/swissknife',
      author='Zeke Arneodo',
      author_email='earneodo@ucsd.edu',
      license='MIT',
      packages=['swissknife'],
      requires=['numpy', 'scipy', 'matplotlib'],
      zip_safe=False)