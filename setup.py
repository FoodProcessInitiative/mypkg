# setup.py の内容例
from setuptools import setup, find_packages

setup(
    name='mypkg', # パッケージ名
    version='0.1.0', # バージョン
    description='A simple utility package for Colab Notebooks',
    author='Yasushi Nakata',
    author_email='yasushi.nakata@omu.ac.jp',
    packages=find_packages(), # 'mypkg'フォルダを自動で見つける
    install_requires=[ # 依存関係があればここに記述
        'numpy>=1.20',
        'matplotlib>=3.3'
    ],
    python_requires='>=3.7',
)
