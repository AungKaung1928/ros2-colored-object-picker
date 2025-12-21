from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'colored_object_picker'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), 
            glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='allkg',
    maintainer_email='aungkaungmyattt1928@gmail.com',
    description='ROS2 package for real-time colored object detection and tracking',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = colored_object_picker.detector_node:main',
        ],
    },
)