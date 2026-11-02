# Windows CPU环境锁

本目录提供Windows x64 / CPython 3.12的依赖版本和wheel SHA-256。锁包含PennyLane、Lightning、PyTorch CPU及科学计算依赖，不包含数据、模型权重或本地安装环境。

在仓库根目录执行，先让`python`指向CPython 3.12 x64，并选择新的环境目录：

```powershell
python -m venv .venv-qbench
.\.venv-qbench\Scripts\python.exe -m pip install --require-hashes -r requirements/locks/qbench-win-cpu-py312-20261028.txt
.\.venv-qbench\Scripts\python.exe -m pip install --no-deps -e .
.\.venv-qbench\Scripts\python.exe -m pip check
New-Item -ItemType Directory -Force outputs | Out-Null
.\.venv-qbench\Scripts\python.exe requirements/locks/validate_qbench_environment.py --output outputs/environment_smoke.json
.\.venv-qbench\Scripts\python.exe -m unittest discover -s tests -v
```

配套`.wheels.json`记录公开下载来源与文件哈希，可用于建立本地离线wheelhouse；离线安装使用`--no-index --find-links <本地目录> --require-hashes`。不要上传wheelhouse或复制venv到另一台机器。

验证脚本只用合成数据，检查`default.qubit/backprop`与`lightning.qubit/adjoint`的前向、梯度、批量预测、保存重载及优化器恢复。它不代替真实模型和数据协议的独立验收。

该锁不适用于Linux、其他Python ABI或GPU/CUDA-Q环境，这些组合需要单独锁定和验证。跨平台、跨后端或跨库版本不保证位级一致。参见[PyTorch可重复性说明](https://docs.pytorch.org/docs/2.6/notes/randomness.html)。
