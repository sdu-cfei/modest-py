from setuptools import setup

setup(name='modestpy',
      version='0.0.5',
      description='FMI-compliant model identification package',
      url='https://github.com/sdu-cfei/modest-py',
      keywords='fmi fmu optimization model identification estimation',
      author='Krzysztof Arendt',
      author_email='krzysztof.arendt@gmail.com',
      license='BSD',
      platforms=['Windows', 'Linux'],
      packages=[
          'modestpy',
          'modestpy.estim',
          'modestpy.estim.ga',
          'modestpy.estim.ps',
          'modestpy.fmi',
          'modestpy.utilities',
          'modestpy.test'],
      include_package_data=True,
      install_requires=[
          'pandas>=0.17.1',
          'matplotlib',
          'numpy>=1.13.0'
      ],
      classifiers = [
          'Programming Language :: Python :: 2.7',
      ]
      )
