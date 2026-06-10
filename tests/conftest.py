import sys
from pathlib import Path

# 把项目根加入 sys.path，使 `import src.xxx` 在测试中可用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
