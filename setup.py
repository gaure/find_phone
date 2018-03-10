from distutils.core import setup

setup(
    name="findPhone",

    # Version number (initial):
    version="1.0.0",

    # Application author details:
    author="gaure",
    author_email="gaure72@gmail.com",

    # Root Packages
    package_dir={'tf_unet': 'tf_unet'},

    # Root Packages
    packages=[""],

    # Include additional files into the package
    include_package_data=True,

    # Details
    url="https://github.com/gaure/find_phone.git",

    # Description
    description="Application to locate a phone in a picture",

    # Dependent packages (distributions)
    install_requires=[
                "pandas",
                "scipy",
                "opencv-python",
                "scikit-learn",
                "Pillow",
                "tensorflow"
            ],
)
