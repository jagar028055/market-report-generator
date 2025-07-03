import os
import sys

# プロジェクトルートディレクトリを追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from .main import main
