# SCALE-Sim with DRAM

将 SCALE-Sim 与 Ramulator2 接通，生成逐层计算周期、DRAM 请求完成时间、带宽与内存 stall 报告。仓库包含两个模拟器及 fmt、yaml-cpp 的完整源码，下载后可以直接构建。

## 快速开始

需要 Linux、Python 3.10–3.12（含 venv 和 pip）、CMake 3.14+，以及支持 C++20 的 GCC 12+ 或 Clang 15+。

```bash
git clone https://github.com/sun123-cmd/scalesim-withDRAM.git
cd scalesim-withDRAM
./run.sh
```

`run.sh` 首次运行会自动创建 `.venv`、安装 Python 依赖、编译 Ramulator2 和回放器，随后运行自带的两层 GEMM。构建默认使用 2 个并行编译任务，C++ 依赖直接使用仓库内的源码。安装 Python 包需要访问包索引。

可单独执行安装或调整构建并行度。

```bash
BUILD_JOBS=4 ./setup.sh
./run.sh
```

通过 `PYTHON=/path/to/python` 和 `CXX=/path/to/compiler` 可以指定解释器与编译器。脚本会自动识别常见编译器及 RHEL GCC Toolset 14。NumPy 固定为 1.26.4，以兼容 SCALE-Sim 的数组到标量转换。

## 查看结果

默认输出目录为 `results/smoke/`。

- `summary.json` 汇总每层基线与 DRAM 延迟反馈后的周期、stall、请求数和完成数。
- `baseline/ramulator2_smoke/COMPUTE_REPORT.csv` 保存基线计算报告。
- `coupled/ramulator2_smoke/COMPUTE_REPORT.csv` 保存反馈后的计算报告。
- `ramulator2/layerN/requests.trace` 保存带时间戳的 DRAM 请求。
- `ramulator2/layerN/responses.csv` 保存每个请求的完成时间。
- `ramulator2/layerN/stats.yaml` 保存 Ramulator2 的控制器和 DRAM 统计。

## 运行自己的模型

```bash
./run.sh \
  -c examples/scale.cfg \
  -t examples/gemm.csv \
  -l examples/layout.csv \
  -i gemm \
  --dram-config examples/ramulator2_ddr4.py \
  --accelerator-mhz 1000 \
  -p results/my_run
```

`-c` 指定 SCALE-Sim 架构配置，`-t` 指定拓扑，`-l` 指定布局。GEMM 使用 `-i gemm`，卷积使用 `-i conv`。`./run.sh --help` 列出所有参数。

默认内存模型采用单通道 DDR4-2400、1 GHz 加速器时钟、每 DRAM 周期最多发射 4 个请求。`--issue-width` 可以调整发射宽度。`--dram-config` 接受定义 `memory_system` 的 Ramulator2 Python 文件，或通过其配置接口导出的完整 YAML。修改示例中的 `controllers` 列表可以设置通道数。

生成的 `feedback.cfg` 也可通过 SCALE-Sim 的标准入口使用。

```bash
.venv/bin/python -m scalesim.scale \
  -c results/smoke/feedback.cfg \
  -t examples/gemm.csv -l examples/layout.csv -i gemm \
  -p results/native_feedback
```

配置项 `Ramulator2LatencyDir` 指定延迟文件目录，相对路径按配置文件所在目录解析。启用该选项时，`InterfaceBandwidth` 需设为 `USER`。

## 建模方式

1. 在 `USER` 模式运行 SCALE-Sim，记录 IFMAP、FILTER、OFMAP 的 DRAM 轨迹。两次 SCALE-Sim 运行使用相同的缓冲区和带宽参数。
2. 还原访存请求的发射时间，将负时间预取区间整体平移至零。每个访存行按 DRAM transaction 大小合并地址，三个流共同进入一个 Ramulator2 内存系统。队列满时重试，全部读写回调完成后结束。
3. 使用 Ramulator2 的 `tCK` 换算加速器周期与 DRAM 周期。按地址出现顺序回填每个 burst 的完成延迟，每个访存行等待其中最晚完成的地址。地址顺序及响应数量均有校验，输出缓冲区可以在 stall 后重新分行。

该接口采用两阶段延迟反馈方法。第二阶段沿用第一阶段测得的 DRAM 竞争与排队延迟，stall 后请求发射时刻的变化会带来近似误差。各层独立初始化 DRAM 状态。写完成沿用 Ramulator2 控制器的回调语义，在终端写命令发出时回调。

地址单位沿用 SCALE-Sim 当前实现的 1 字节 word。`-1` 作为 padding，地址 `0` 正常参与仿真。同一行、同一 transaction 内的地址合并，跨行 refetch 保留。大于 10000 周期的延迟完整回填。

## 验证

```bash
.venv/bin/python -m pytest \
  SCALE-Sim/test/test_ramulator2.py \
  SCALE-Sim/test/test_read_buffer_stall_accounting.py -q
```

10 项测试覆盖 burst 合并、地址零值、padding、乱序完成、时钟边沿对齐、大延迟、重新分行和缺失响应。端到端验证覆盖 WS 单通道 GEMM、OS 双通道 GEMM、IS 卷积。512 请求背压测试验证完成 ID、请求数量、损坏输入和周期上限。标准 CLI 的基线及反馈报告分别与集成入口一致。

## 源码与许可证

- `SCALE-Sim/` 来自 scalesim-project/SCALE-Sim 的 `87fd03d4ab3ac800119fc88904440c6da420226b`，包含此次集成改动。
- `ramulator2/` 来自 CMU-SAFARI/ramulator2 的 `7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da`，上游版本为 2.1。
- `ramulator2/ext/fmt/` 固定在 `e69e5f977d458f2650bb346dadf2ad30c5320281`。
- `ramulator2/ext/yaml-cpp/` 固定在 `56e3bb550c91fd7005566f19c079cb7a503223cf`。

各项目的许可证随源码保留。集成入口为 `SCALE-Sim/scalesim/ramulator2.py`，延迟端口为 `SCALE-Sim/scalesim/memory/ramulator2_port.py`，C++ 回放器为 `integration/replay.cpp`。
