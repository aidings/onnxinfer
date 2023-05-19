from setuptools import find_packages, setup

setup(name='onnxinfer',  # 包名
      version='0.0.1',  # 版本号
      description='',
      long_description='',
      author='aidings',
      author_email='zhifeng.ding@hqu.edu.cn',
      url='https://github.com/aidings/onnxinfer.git',
      license='',
      install_requires=['opencv-python', 'numpy', 'pillow', 'onnxruntime-gpu'],
      extras_require={},
      dependency_links=[
          "https://pypi.tuna.tsinghua.edu.cn/simple",
          "http://mirrors.aliyun.com/pypi/simple"
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7'
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Utilities'
      ],
      keywords='',
      packages=find_packages('src', exclude=["examples", "tests", "project"]),  # 必填
      package_dir={'': 'src'},  # 必填
      include_package_data=True,
      scripts= [
      ],
)