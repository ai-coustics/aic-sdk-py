import glob
import os

from setuptools import find_packages, setup

# Find all binary files that should be included
package_data = {}
for root, dirs, files in os.walk("aic/libs"):
    for file in files:
        if file.endswith(('.so', '.dylib', '.dll')):
            rel_dir = os.path.relpath(root, "aic")
            if "aic" not in package_data:
                package_data["aic"] = []
            package_data["aic"].append(os.path.join(rel_dir, file))

setup(
    name="aic-sdk-py",
    version="0.5.3",
    description="Python bindings for the ai|coustics speech-enhancement SDK",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="ai-coustics GmbH",
    author_email="info@ai-coustics.com",
    license="Apache-2.0",
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    # This is crucial - it tells setuptools this is not a pure Python package
    zip_safe=False,
    has_ext_modules=lambda: True,  # Force platform wheel
) 