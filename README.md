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

## 在线闭环协同仿真与搜索

```bash
./run_online.sh prepare-costs
./run_online.sh compile --cores 32 --target-gbps 136.6 --output results/online_program
./run_online.sh simulate --cores 32 --target-gbps 136.6 \
  --kp 2 --kd 2 --batch 2 --replicas 1 --max-inflight 64 \
  --output results/online_136
```

`prepare-costs` 通过 SCALE-Sim 的 weight-stationary 计算入口测量实际边界 tile 的 cycle 数并缓存。`compile` 生成完整 π0.5 lazy tile 程序、全局地址布局与目标内存配置。当前负载采用 **horizon 16、10 个 denoise timestep、968 个 prefix token、BF16**。执行内容为符号 GEMM 的计算时序、DRAM 地址请求和完成事件，输出包含架构周期、延迟、带宽和请求统计。

`simulate` 在一个原生 C++ 事件循环中运行全部 stage 和副本。每个 lane 使用双输入 buffer，tile 的全部输入返回后开始计算，计算周期由代价表查得，C 累加器跨 K tile 保留，最终输出发往 DRAM。stage 的 DMA 请求、Ramulator2 回调、微批就绪、timestep 前进和交接均在该循环内推进。每个候选执行一次完整在线仿真；无争用 stage 参考值作为独立缓存条目建立。

每核私有 SRAM 默认 1 MiB，阵列为 32×32。`--core-sram-mib 0.25|1|4` 和 `--shared-sram-mib 0|64|256` 覆盖计划中的九种 SRAM 配置。共享 SRAM 按总核数等分，并作为每 lane 的额外 tile 容量，当前采用统一容量模型，未另行模拟共享 SRAM 的 bank 和端口竞争。M tile 根据双缓冲 BF16 A/B 与 FP32 C 的总占用选择，N/K tile 为 32。DiT microbatch 沿 dense GEMM 的 M 维合并，attention 保留每请求的独立 KV。编译器展开实际 transformer 层顺序，Q/KV/attention/MLP 的权重按层偏移引用同一份全局区域，Q→QK→AV→输出及 gated MLP 使用真实 tensor view 地址别名。

在线模式的 `--max-inflight` 表示每个 stage 的在途 DMA transaction 上限，涵盖该 stage 的全部 lane。输入按饱和处理，stage 输入队列和可循环使用的请求 KV slot 产生物理反压。权重地址在 stage、副本和 timestep 之间共享，每个在飞请求的 prefix/suffix KV 占独立区域，完成后回收 slot。stage-local activation scratch 按当前活跃作业复用。handoff 通路按全局 256 GB/s 串行传输就绪作业。

默认先完成 20 个预热请求，再连续选择 50 个完成请求计算平均/p95 L 和平均输出间隔 S。L 从请求 admission 起算，包括流水内部等待和凑批时间。stage 报告记录 DMA 在途峰值、读延迟、实际带宽、row-hit rate、访存停顿和完整作业的 service time。`--events` 可保存 tile 读写、计算和交接的 JSONL 事件日志。每个候选目录包含 `program.yaml`、`native.yaml`、`native.log` 和 `result.json`，顶层 `summary.json` 汇总单方案结果。

### 目标带宽与内存配置

```bash
./run_online.sh simulate --target-gbps 819 --cores 128 \
  --kp 2 --kd 5 --batch 5 --dit-core-numerator 8 \
  --output results/online_819
```

目标 136.6 GB/s 使用 2 个 64-bit LPDDR5 控制器、tCK=937 ps，配置峰值为 136.6062 GB/s，作为 LPDDR5X 带宽目标的 LPDDR5 时序近似。目标 819 GB/s 使用真实 HBM3 协议与 HBM34 控制器，共 16 个 64-bit channel，每个含两个 pseudochannel。沿用上游 HBM3_6400Mbps preset，名义峰值 819.2 GB/s，整数 ps 半时钟量化后的模型峰值为 820.5128 GB/s。实际有效带宽由访问模式和共享竞争决定。

LPDDR5 时序在新时钟下按绝对 ns 时序向上取整。HBM3 沿用上游 preset，其中部分 core timing 在上游注明为估计值。`memory_metadata` 保存协议、requested/effective bandwidth、控制器数、tCK、全部 timing overrides 和来源。两档物理容量和逻辑地址预算均为 16 GiB。

### 每硬件点分支定界

```bash
./run_online.sh search --cores 32 --target-gbps 136.6 \
  --workers 150 --batch-size 150 --objective pareto \
  --output results/search_32_136
```

每个硬件点独立枚举合法的 prefix/timestep stage、microbatch 和副本组合，独立枚举每个 prefix stage 和 DiT stage 的阵列数 1/2/4/8/16/32/64/128，允许未分配阵列空闲。单候选可通过 `--prefix-stage-cores` 和 `--dit-stage-cores` 同时指定。共享 DRAM 候选使用统一竞争策略。顺序基线覆盖 batch 1/2/5/10 与副本 1/2/4。`--include-uneven` 增加 3 和 4 个 DiT stage 的 timestep 分区。

候选按乐观 S、L 排序，每批最多 150 个由独立原生进程执行，维护实际 Pareto 前沿和吞吐最优 S*。`--objective throughput` 使用吞吐停止条件，`--objective pareto` 使用剩余乐观点的支配条件。达到主要停止条件后，搜索继续审核可能进入 2% 等价集合与 S≤1.1×S* 延迟最优集合的候选。`search_summary.json` 输出前沿、两类最优点、2% 集合、评估数量和停止证书。每个实际评估方案含逐 stage 的实际/无争用 service-time 比值。

默认 `--bound-source certified --error-margin 0` 使用实测 tile compute work、已分配 lane 数和有限测量窗口 burst 上限构造下界。无争用 native stage service-time 表通过一次完整 stage job 建立，按程序、地址、内存和引擎版本签名缓存，并用于争用减速报告。

按 pilot 的层可加性误差使用无争用表估计排序与剪枝，可直接生成并读取实测 e。

```bash
./run_online.sh pilot --cores 32 --target-gbps 136.6 \
  --kp 2 --kd 2 --batch 2 --output results/search_pilot_bounds
./run_online.sh search --cores 32 --target-gbps 136.6 \
  --bound-source uncontended-stage-table \
  --pilot-report results/search_pilot_bounds/pilot.json \
  --workers 150 --batch-size 150 --output results/search_pilot_bounds
```

`pilot` 将实际配置的每个 stage 作为完整作业独立运行，再按原有顺序切成 transformer layer 作业独立运行。报告逐 stage 的 `abs(sum(layer services) - whole stage service) / whole stage service`，取最大值作为该 pilot 的 e。`pilot.json` 保存所有原生参考报告与模型签名，搜索检查运行模型和硬件点一致。多个 pilot 配置的最大误差也可通过 `--error-margin` 传入。

该模式的证书依赖已测 pilot 误差对候选空间的适用性，结果标记为 conditional。开发时可用 `--candidate-limit`、`--max-evaluations` 和 `--timeout-seconds` 限定预算，结果会记录候选空间与评估的完整性。`checkpoint.json` 支持同一执行配置续跑，候选和参考表缓存按配置签名验证。并行数应与实际分配的 CPU 和内存资源一致。

### 在线模式验收范围

原生验收使用真实 Ramulator2 和小型完整十 timestep 程序，检查回调到计算的因果关系、最终 K 累加后的写回、双缓冲和 DMA 限额、batch 权重共享、KV/副本隔离、target 控制器配置、顺序基线和有限队列反压。编译器验收覆盖完整 π0.5 的形状、MAC 守恒、代价表、SRAM 占用、共享权重及 tensor view 地址。搜索验收将剪枝结果与小型空间穷举结果对照，并验证缓存、续跑和等价集合完整性。

完整模型具有真实尺寸和逐 transaction 请求量，其 70 请求运行及全候选搜索的耗时取决于配置和实际资源。完成完整运行的结果以相应 `native.yaml`、`result.json` 和搜索证书为准。

```bash
.venv/bin/python -m pytest integration/check_online_*.py -q
ONLINE_BINARY=integration/build/ramulator2-online \
  .venv/bin/python -m pytest integration/check_online_native.py -q
```

## Proposal 解析与校准时间线入口

当前实验统一使用 **action horizon 16、denoise steps 10、prefix tokens 968、BF16**。阵列为 32×32、1 GHz，每核私有 SRAM 总量为 1 MiB，共享 SRAM 为 64 MiB。目标 DRAM 容量为 16 GiB，带宽按 LPDDR5X 目标分别为 68.3、136.6、273.2、546.4 GB/s。

```bash
./run_proposal.sh --beta auto
./run_proposal.sh --all-hardware --beta auto --output results/proposal_16
./run_proposal.sh --kp 3 --kd 5 --batch 5 --dit-core-numerator 4 --beta auto
```

默认代表点为 32 核、2 个逻辑 channel、2 个 prefix stage、2 个 DiT timestep stage、b=2、1 个副本、DiT 分配 4/16 核心。16 个硬件点与 proposal 表格一致。3 和 4 个 DiT stage 分别分配 timestep 为 4/3/3、3/3/2/2。prefix 按真实连续层进行计算量均衡切分。

200 个请求按饱和 admission 输入，前 50 个和最后 20 个完成请求用于裁剪。`summary.json` 输出平均/p95 L、有效输出间隔 S、各 stage 成本、四种 batch 的整片顺序基线；`timeline.json` 保存每个请求和 timestep 的时间线。多硬件入口输出 `hardware_summary.csv` 和 JSON，无效组合会记录原因后继续。

L 从请求进入 prefix stage 0 起计，包含凑批和内部排队。有限 WIP credit 默认每副本为 kp+b×(kd+1)，可通过 `--max-inflight` 指定。交接总带宽 256 GB/s 在全部副本的 stage 边界间均分，microbatch 的交接字节按 b 增长。时间线检查 stage 资源互斥和 timestep 顺序，报告带窗口边界修正的 Little's law 残差及 `steady_state_flag`。深流水和大 batch 的 200 请求运行可能尚未稳定，应保留该标记并增加请求数验证。

理想访存账本支持 camera dense projection 权重共享、DiT microbatch 权重共享和 MQA 的 KV-head 复用。horizon16 时，每请求 prefix 精确 FLOPs 为 4,635,481,079,808，DiT 每 timestep 为 12,526,288,896，10 timestep 后总量为 4,760,743,968,768。权重账本分别为 4,791,831,552 B 和 859,963,392 B，KV 为 17,842,176 B/request。

### SCALE-Sim 校准与共享队列诊断

```bash
./run_proposal.sh --beta auto --calibrate-tiles --profile-memory \
  --memory-policy both --output results/proposal_pilot
```

SCALE-Sim 校准使用 32×32 阵列、每核总计 1 MiB 私有 SRAM，三个 operand buffer 分别为 384/384/256 KiB。对唯一分块测量计算周期，以每 stage 的阵列数计算分块波次，结合理想字节账本与静态带宽计算 stage 时间。`calibration.json` 保存可核对的分块结果。共享 SRAM 的配置与 stage 份额会记录。固定小分块校准主要测量阵列填充与排空，完整算子的 SRAM 敏感性及跨算子共享 SRAM 复用属于后续校准范围。

原生回放器接受第五列 `stage_id`，将所有 stage 的请求送入同一个内存系统。`shared` 使用统一竞争队列及轮询发射；`static` 使用逐 stage 带宽限速。输出每 stage 的实际带宽、配额、读延迟、row-hit rate、重试次数，以及总带宽和物理峰值利用率。旧的四列轨迹和调用入口继续支持。

此解析入口的原生内存诊断使用 LPDDR5-6400 x16 代理，每物理控制器峰值为 12.8 GB/s、容量为 1 GiB。目标预算与代理物理参数分别保存，L/S 来自解析 stage 成本和时间线。完整请求的在线闭环、目标带宽配置与分支定界搜索使用 `run_online.sh` 入口。NeuSim 对照与完整三轮筛选实验可沿用这一解析入口进行交叉验证。

计划中 32 核、DiT 分配 8 核、kd=10 的 pilot 组合会触发资源不足校验。单 prefix stage、单 DiT stage 的流水具有独立资源，整片顺序基线依次占用资源，二者的退化关系需要在资源一致的条件下校验。

回归测试可执行以下命令。原生队列测试通过环境变量指定已导出的内存配置。

```bash
.venv/bin/python -m pytest integration/check_*.py \
  SCALE-Sim/test/test_ramulator2.py SCALE-Sim/test/test_read_buffer_stall_accounting.py -q
REPLAY_CONFIG=results/proposal_pilot/memory_profile/memory.yaml \
  .venv/bin/python -m pytest integration/check_shared_replay.py -q
```

## π0.5 独立分块估算入口

```bash
./run_pi05.sh
```

默认采用官方 OpenPI π0.5 的 BF16、3 路 224×224 图像、200 个 padded prompt token、16 步动作 horizon、10 次去噪迭代和 batch=1。模型形状固定于 OpenPI 提交 `215abfb217dbac7d5f1273282331b9b1866c0479`，源码链接保存在生成的 manifest 中。

入口生成完整尺寸的 26 个 GEMM 算子族，覆盖 SigLIP 视觉编码器、PaliGemma prefix KV、动作专家、时间投影和 AdaRMS 调制投影。所有 M/N/K 维度按 32 分块，真实边界尺寸保留，唯一分块模拟一次，再按完整模型的层数、head、相机和去噪次数汇总。默认全结构共 `2,380,371,984,384` MAC，实际分块计数和完成请求数记录在随附验证摘要中。

`results/pi05/summary.json` 包含逐算子、逐阶段与全结构的 MAC、分块数、加权周期、stall 和 DRAM 请求计数。`results/pi05/tile_simulation/` 保存实际模拟的分块报告和逐请求轨迹。`examples/pi05/validated_summary.json` 保存一次验证摘要。

估算范围为 GEMM 算子与 DRAM 访存。采用独立冷分块、串行成本相加和两阶段 DRAM 延迟回填，跨分块 SRAM/权重复用与算子重叠仍待建模。K 分块每次写出输出，其部分和读取与加法流量仍待建模。归一化、softmax、RoPE、激活函数、embedding gather 和 Euler 更新属于后续覆盖范围。加权周期用于该建模条件下的架构比较。

可调整分块尺寸与 DRAM 配置。

```bash
./run_pi05.sh --tile-size 64 --output results/pi05_tile64
```

可单独生成其他输入尺寸的完整结构，再执行模拟。

```bash
.venv/bin/python integration/pi05_model.py \
  --cameras 2 --prompt-tokens 128 --action-horizon 50 \
  --denoise-steps 10 --batch-size 1 --output results/pi05_custom.json
.venv/bin/python integration/pi05_runner.py \
  --manifest results/pi05_custom.json --output results/pi05_custom
```

`WordSizeBytes` 与 `--word-bytes` 支持 1、2、4、8 字节 word，同时作用于 SRAM 元素容量、DRAM 字节地址与完成延迟映射。π0.5 默认为 2 字节 BF16。

适配及分块守恒测试可一并执行。

```bash
.venv/bin/python -m pytest integration/check_pi05.py integration/check_pi05_model.py \
  SCALE-Sim/test/test_ramulator2.py SCALE-Sim/test/test_read_buffer_stall_accounting.py -q
```

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

## 轨迹回放入口的建模方式

1. 在 `USER` 模式运行 SCALE-Sim，记录 IFMAP、FILTER、OFMAP 的 DRAM 轨迹。两次 SCALE-Sim 运行使用相同的缓冲区和带宽参数。
2. 还原访存请求的发射时间，将负时间预取区间整体平移至零。每个访存行按 DRAM transaction 大小合并地址，三个流共同进入一个 Ramulator2 内存系统。队列满时重试，全部读写回调完成后结束。
3. 使用 Ramulator2 的 `tCK` 换算加速器周期与 DRAM 周期。按地址出现顺序回填每个 burst 的完成延迟，每个访存行等待其中最晚完成的地址。地址顺序及响应数量均有校验，输出缓冲区可以在 stall 后重新分行。

该接口采用两阶段延迟反馈方法。第二阶段沿用第一阶段测得的 DRAM 竞争与排队延迟，stall 后请求发射时刻的变化会带来近似误差。各层独立初始化 DRAM 状态。写完成沿用 Ramulator2 控制器的回调语义，在终端写命令发出时回调。

地址以 word 为单位，默认每 word 为 1 字节，可通过 `WordSizeBytes` 设置。`-1` 作为 padding，地址 `0` 正常参与仿真。同一行、同一 transaction 内的地址合并，跨行 refetch 保留。大于 10000 周期的延迟完整回填。

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
