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
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    # This is crucial - it tells setuptools this is not a pure Python package
    zip_safe=False,
    has_ext_modules=lambda: True,  # Force platform wheel
) 