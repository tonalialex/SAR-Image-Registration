# SAR 图像两轮复原与配准：完整代码

这是已在服务器验证的最新版统一流程代码，包含两轮 NAFNet 复原、三层 SAR-SIFT 前端、SimCLR 训练及迭代配准。

## 最新规则

- NCC 数值仅作报告，不作为候选变换或最终验收的拒绝门槛。
- 报告中的 `RMSEloo` 为真正的留一预测 RMSE：逐对移除保留匹配点，用其余点重新拟合仿射变换，计算被移除点的预测残差，再汇总为 RMSE。学习匹配沿用置信度权重，SAR-SIFT 阶段使用等权重；点数不足或留一拟合退化时报告为 `null`。`RMSEloo_method` 标识该定义，历史报告需重新生成才能使用新定义。
- 空间筛选后不足最低匹配数、但筛选前几何内点足够时，使用筛选前全部几何内点。统一入口最低要求为 6 对。
- 回退时跳过空间覆盖及数量预算门槛，保留真实目标归属、仿射合法性、匹配误差和预测留一误差检查。

## 运行

建议在有 CUDA 的服务器上运行。Python 3.12，服务器已验证的 PyTorch 构建为 2.5.1+cu124；根据运行机器先安装适合该机器的 GPU 版 PyTorch，再安装其余依赖。

```bash
python -m pip install -r requirements.txt
python run_full_pipeline.py --input-dir /path/to/inputs --output-dir /path/to/new_output --expected-pairs 7 --device cuda --workers 3
```

单对图像：

```bash
python run_full_pipeline.py --reference /path/scene_active.jpg --sensed /path/scene_passive_PGA64.jpg --output-dir /path/to/new_output --device cuda
```

输入文件同一前缀成对出现：active 为参考图，passive 或 passive_PGA数字 为待配准图。输出目录必须不存在。路径参数可改成 Windows 路径；有空格时加引号。默认两轮复原各 300 步，描述子训练 100 epochs，最多 4 次匹配迭代。

## 文件

- `run_full_pipeline.py`：唯一推荐的完整流程入口。
- `restoration/`：两轮复原代码及所需模块；`third_party/NAFNet/` 保留网络实现与许可证。
- `registration/`：最新版配准代码、指标计算与回归检查。
- `models/simclr_initial.pt`：统一入口必需的初始化权重，来源为原项目既有初始化依赖；不是本轮某对图像训练后的模型。

复原和配准通过独立子进程运行，避免两个同名 `sar_registration` 包相互导入。复原包内的配准模块是复原依赖快照；最新版后续配准由 `registration/` 包执行。

该目录只包含完整源代码、必需初始化权重、许可证和使用说明，不包含数据集、实验图像、指标、训练日志或图对训练模型。此次整理未运行新的真实图像实验。
