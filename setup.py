from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'elsabot_speech_input'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'wakewords'), glob('wakewords/*')),
    
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Scott Horton',
    maintainer_email='none@none.com',
    description='Elsabot Speech Input Processing Node',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'elsabot_speech_input = elsabot_speech_input.node:main',
        ],
    },
)
