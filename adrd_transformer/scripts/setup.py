import setuptools

setuptools.setup(
    name = 'adrd',
    version = '0.0.1',
    author = 'Chonghua Xue',
    author_email = 'cxue2@bu.edu',
    # url = 'https://github.com/CRANK-MS/CRANK-MS',
    # description = 'Re-implementation of Python Speech Features Extraction on CUDA.',
    packages = setuptools.find_packages(),
    package_data = {'adrd': [
        'ckpt/ckpt_080823.pt',
        'ckpt/ckpt_img_072523.pt',
        'ckpt/dynamic_calibrated_classifier_073023.pkl',
        'ckpt/static_calibrated_classifier_073023.pkl',
    ]},
    python_requires = '>=3.11',
    classifiers = [
        'Environment :: GPU :: NVIDIA CUDA',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    install_requires = [
        'numpy',
        'torch',
        'tqdm',
        'scikit-learn',
    ],
)
