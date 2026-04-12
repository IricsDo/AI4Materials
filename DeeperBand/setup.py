import os
import re
from setuptools import setup, find_packages

require_file = 'requirements.txt'
package_name = "deeperband"

# ==================== Version extraction ====================
verstrline = open(os.path.join(package_name, '__init__.py'), 'r', encoding='utf-8').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError(f'Unable to find version string in "{package_name}/__init__.py".')

# ==================== Requirements ====================
with open(require_file, encoding='utf-8') as f:
    requirements = []
    for line in f.read().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            req = line.split('#')[0].strip()
            if req:
                requirements.append(req)

# ==================== Long description ====================
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Li Jun',
    author_email='ljcj007@ysu.edu.cn',
    url='https://github.com/ljcj007/DeeperBand',
    python_requires='>=3.10',          # adjust if needed
    entry_points={
        'console_scripts': [
            f'{package_name} = {package_name}:main'
        ]
    },
)