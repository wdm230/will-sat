from setuptools import setup, find_packages

# Dependencies are managed via environment.yml using Conda; install_requires is left empty.
setup(
    name="mlc_pipeline",
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
