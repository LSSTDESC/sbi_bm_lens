from setuptools import find_packages, setup

setup(
    name="sbiax",
    version="0.0.1",
    url="https://github.com/LSSTDESC/sbi_jax",
    description="JAX-based sbi package",
    packages=find_packages(),
    package_dir={"sbiax": "sbiax"},
    package_data={
        "sbiax": ["data/*.csv", "data/*.npy", "data/*.pkl"],
    },
    include_package_data=True,
    install_requires=[
        "sbi_lens @ git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git",
        "numpy>=1.22.4,<1.24",
        "jax>=0.4.1",
        "tensorflow_probability>=0.19.0",
        "dm-haiku>=0.0.9",
        "jaxopt>=0.6",
        "numpyro>=0.10.1",
        "jax-cosmo>=0.1.0",
        "optax>=0.1.4",
        "wheel",
    ],
)