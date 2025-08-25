from setuptools import setup, find_packages

HYPHEN_E_DOT = "-e ."

def get_requirements(filepath: str):
    with open(filepath) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        # Remove "-e ." if present
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements




setup(
    name="visa_mlproject",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    author="Srinu",
    author_email="srinunayakk7@gmail.com",
    install_requires=get_requirements("requirements.txt"),
)