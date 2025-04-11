# setup.py
from setuptools import setup, find_packages

setup(
    name="mlc_pipeline",
    version="0.1.0",
    description="A modular processing pipeline for Sentinel-2 composites, DEM retrieval, classification, and meshing.",
    author="Will McKelvey",
    author_email="william.d.mckelvey@usace.army.mil",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
        "matplotlib",
        "Pillow",
        "trimesh",
        "requests",
        "earthengine-api",
        "triangle",
        "scikit-learn",
        "Shapely",
        "pyproj",
        "pyyaml",
        "scikit-image"
    ],
    entry_points={
        "console_scripts": [
            "run-mlc=mlc_pipeline.pipeline:main"
        ],
    },
    zip_safe=False,
)
