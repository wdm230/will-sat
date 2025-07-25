from setuptools import setup, find_packages


setup(
    name="will-sat",
    version="1.0.0",
    description="A modular processing pipeline for Sentinel-2 composites, DEM retrieval, classification, and meshing.",
    author="Will McKelvey",
    author_email="william.d.mckelvey@usace.army.mil",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "run-mlc=mlc_pipeline.pipeline:main"
        ],
    },
    zip_safe=False,
)
