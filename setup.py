from setuptools import find_packages, setup

package_name = 'stereo_processor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Janith-Chamikara',
    maintainer_email='janithchamikara13@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'calibration_capture = stereo_processor.calibration_capture:main',
            'stereo_sgbm_node = stereo_processor.stereo_sgbm_node:main',
            'stereo_sync_capture = stereo_processor.stereo_sync_capture:main',
            'stereo_cuda_bm_node = stereo_processor.stereo_cuda_bm_node:main',
            'stereo_cpu_tuning_node = stereo_processor.stereo_cpu_tuning_node:main'
        ],
    },
)
