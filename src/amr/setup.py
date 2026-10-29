from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'amr'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
        (os.path.join('share', package_name, 'sdf'),
            glob('sdf/*')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*')),
        (os.path.join('share/', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share/', package_name, 'map'),
            glob('map/*')),
        (os.path.join('share/', package_name, 'planner'),
            glob('planner/*')),
        (os.path.join('share/', package_name, 'costmaps'),
            glob('costmaps/*.yaml')),
        (os.path.join('share/', package_name, 'bt'),
            glob('bt/*.xml')),
        (os.path.join('share/', package_name, 'controller'),
            glob('controller/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='khalandar-workspace',
    maintainer_email='dadakhalandarshk@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={    
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'map_reader = planner.map_reader:main',
            'yolo_dic = nodes.yolo:main',
            'yolo_world_dic = nodes.yolo_world:main',
            'image_collecting = nodes.image_collecting:main',
        ],  
    },
)