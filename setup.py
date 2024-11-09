from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'My first chat bot '
LONG_DESCRIPTION = 'Understanding how the NLP works'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="chatbot", 
        version=VERSION,
        author="Ben",
        author_email="",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        package_dir={'':'src'},
        packages=find_packages(where='src'),
        install_requires=['scikit-learn', 'nltk', 'gensim'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'

        keywords=['python', 'chat'],
        # classifiers= [
        #     "Development Status :: 3 - Alpha",
        #     "Intended Audience :: Education",
        #     "Programming Language :: Python :: 2",
        #     "Programming Language :: Python :: 3",
        #     "Operating System :: MacOS :: MacOS X",
        #     "Operating System :: Microsoft :: Windows",
        # ]
)
