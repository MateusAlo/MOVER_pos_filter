from setuptools import find_packages, setup

package_name = 'ps_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ps_node_launch.py']),
        ('share/' + package_name + '/config', ['config/ps_node_params.yaml']),
        ('share/' + package_name + '/scripts',['scripts/fake_data_publisher.py'])

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
            "ps_node = ps_localization.ps_node:main",
            "fake_data_publisher = scripts.fake_data_publisher:main"
        ],
    },
)
