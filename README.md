# AI4CT

放疗计划质量保证（PSQA）研究代码：提取 DICOM RT Plan 复杂度特征，比较经典机器学习与量子经典混合模型，并验证训练、预测和模型重载流程。

**公开仓库只包含代码、合成测试、通用方法说明和依赖锁。** 原始DICOM、Excel/JSON标签、特征表、患者映射、实际队列配置、训练结果、模型权重、Notebook输出和原始文献/报告均保留在本地，不随仓库发布。这里不提供真实数据准确率或可下载的患者数据。

## 代码结构

```text
src/ai4ct/
  paths.py          # 项目、数据与输出路径
  io/               # DICOM读取
  features/         # 历史特征实现；不同定义不可直接混用
  reproduction/     # 显式映射、单层特征、论文适配与来源差异分析
  qbench/           # 经典/混合模型、训练、checkpoint与独立审计
  benchmarks/       # 历史基线入口
  advanced/         # 历史深度模型入口
tests/              # 运行时生成合成样本；不包含真实患者文件
configs/public/     # 算法资料和模型参数，不包含实际队列manifest
requirements/locks/ # Windows CPU依赖版本、wheel哈希及合成验证
docs/public/        # 使用说明、研究边界与第三方来源
examples/           # 独立Transformer教学代码
```

方法范围和原文差异见[算法名录](docs/public/algorithms.md)、[研究边界](docs/public/research_boundaries.md)。[模型配置目录](configs/public/model_configurations.json)仅保存实现参数，不能直接作为训练manifest执行。不同编码、消融和经典控制不等于独立算法家族。

## 安装与合成测试

核心包要求Python 3.10+；锁定环境为Windows x64 / Python 3.12。PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[reproduction,qbench]"
python -m unittest discover -s tests -v
python -m ai4ct --help
```

精确依赖复建使用[环境锁说明](requirements/locks/README.md)。GPU、CUDA-Q或其他平台需要另外配置与验证；依赖声明不保证任意组合都已验证。完整模型测试需要Torch和PennyLane，部分可选后端未安装时测试可能跳过，请检查skip记录。

## 使用自己的本地数据

1. 将原始计划和标签存放于本地`data/raw/`，建立有证据的逐计划QA对应关系与患者分组。数量或行号相同不构成对应证据。
2. 使用`ai4ct.reproduction`建立本地特征、来源和固定分组折分。标准化、特征选择等学习步骤限定在训练折内。
3. 在本地`configs/local/`保存实际队列配置，明确特征版本、标签阈值、划分及输入哈希。此目录默认不参与Git。
4. 每次运行选择新的`outputs/`子目录，保存模型、逐样本预测和审计记录，不覆盖原始数据或历史结果。

`AI4CT_PROJECT_ROOT`、`AI4CT_DATA_DIR`、`AI4CT_OUTPUT_DIR`支持绝对路径覆盖；配置导入不创建文件。以下入口只显示帮助：

```powershell
python -m ai4ct doctor --help
python -m ai4ct.reproduction.prepare_features --help
python -m ai4ct.reproduction.run --help
python -m ai4ct.qbench.runner --help
python -m ai4ct.qbench.audit --help
python -m ai4ct.reproduction.cohort_shift --help
```

没有私有数据的新克隆运行`doctor`会报告输入缺失，这是预期的数据准备状态。历史`benchmarks/advanced`入口保留用于追溯，部分流程仍有研究设计局限，不能直接用于已验证的性能结论。数据来源分类与QA预测的区别见[研究边界](docs/public/research_boundaries.md)。

独立教学示例的帮助入口是`python -m examples.transformer_imdb.run --help`；实际运行可能下载公开IMDB资料，和PSQA流程无关。

## 公开内容检查

`.gitignore`采用发布目录白名单。提交前执行：

```powershell
git add .
python scripts/check_public_release.py --staged
git diff --cached --stat
```

检查器拒绝数据目录、模型文件、二进制附件、个人路径和常见密钥格式。当前公开版本不启用GitHub Actions，请在提交前手动执行检查。自动检查不能替代人工审阅，不应使用`git add -f`绕过排除规则。原有本地研究文档和结果保持原位置，不会因未上传而删除。

[第三方代码与许可说明](docs/public/third_party.md)。本项目没有额外授予统一的开源许可证；第三方文件各自保留原许可证和版权声明。
