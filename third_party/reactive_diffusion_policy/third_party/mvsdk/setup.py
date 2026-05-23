from setuptools import setup, find_packages

setup(
    name="mvsdk",
    version="1.0",
    keywords=("mvsdk"),
    description="Mindvision Python SDK",
    long_description="Mindvision SDK for Python",
    license="MIT Licence",

    # packages = find_packages(),  
    packages=['mvsdk'],
    package_data={
        'mvsdk': ['lib/*/*.so']
    },
    # platforms="linux",
    install_requires=[],
    #
    # scripts = [],
    # entry_points = {
    #     'console_scripts': [
    #         'test = test.help:main'
    #     ]
    # }
)
