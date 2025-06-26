from setuptools import setup

package_name = 'simple_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Simple camera publisher',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = simple_camera.camera_publisher:main',
            'camera_subscriber = simple_camera.camera_subscriber:main',
            'simple_publisher = simple_camera.simple_publisher:main',
        ],
    },
)
