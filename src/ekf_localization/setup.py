from setuptools import find_packages, setup

package_name = 'ekf_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ekf_localization_launch.py']),
        ('share/' + package_name + '/config', ['config/navsat_transform.yaml', 'config/ekf_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mateus',
    maintainer_email='mateusalonso@usp.br',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_node = ekf_localization.ekf_node:main',
        ],
    },
)
