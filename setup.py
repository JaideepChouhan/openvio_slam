from setuptools import setup, find_packages

package_name = 'openvio_slam'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['tests*', 'tools*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/openvio_slam.launch.py']),
        ('share/' + package_name + '/config', ['config/slam.yaml']),
    ],
    install_requires=['numpy', 'scipy', 'opencv-python'],
    entry_points={
        'console_scripts': [
            'openvio_slam_node = openvio_slam.ros.vio_node:main',
        ],
    },
    zip_safe=True,
    author='OpenVIO-SLAM Contributors',
    description='From-scratch monocular VIO-SLAM for indoor ground robots',
    license='MIT',
    python_requires='>=3.8',
)
