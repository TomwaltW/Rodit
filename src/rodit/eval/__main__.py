"""python -m rodit.eval：在合成评测集上跑演示单元（默认离线替身；--llm env 接真实大模型）。

选项、输出和退出码见 rodit.eval.cli。
"""

from rodit.eval.cli import main

raise SystemExit(main())
