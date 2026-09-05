# scripts/cluster —— 集群（多 GPU 节点）验证工具

本目录随 repo 入库：**集群上 `git pull` 一次即可获得全部工具与 tests/，无需再手工 scp**。
probe 脚本均按"自身所在目录"定位（job 脚本内 `HERE=` 解析），不依赖手工摆放位置。

| 文件 | 用途 |
|---|---|
| `job_validate_all.sh` | 统一验证作业：全量 pytest + 三档（auto/float/double）数值一致性 + mem_probe 显存 + perf_probe 性能 + 可选真实分析 CONFIG 单步 NLL。SLURM 双卡（v100:2） |
| `mem_probe.py` | 分阶段显存探针（config 解析/积分/A/hessian），物理卡 nvidia-smi 交叉核对 |
| `perf_probe.py` | 各段耗时/显存基准（NLL ×10 / Hessian ×3） |
| `ff_eff_probe.py` | float 档 FF/EFF/writeResult 冒烟（单测已并入 test_float_mode.py，保留作独立工具） |
| `sync_to_cluster.sh` | 后备通道（集群无法访问 GitHub 时）：pack/push/pull 文件级同步；`pull` 模式可拉回 `log_*.out` |

## 集群一次性设置（把已有目录变成 git 仓库）

集群 CTPWA_ROOT（`/hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa`）如已是手工同步的文件拷贝，转换：

```bash
cd /hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa
git init . && git remote add origin git@github.com:BHXiang/ctpwa.git
git fetch origin
git diff origin/main --stat          # 先看本地与 GitHub 的差异（本地未入库修改会在此列出）
git checkout -f -B main origin/main  # 已跟踪文件以 GitHub 为准；.so/build/ 等 untracked 保留
```

> 注意：`checkout -f` 只覆盖**已跟踪**文件（源码/tests），不删除 untracked（`ctpwa.so`、
> `build/`、日志、`_gen_*.yml` 等运行产物都保留）。若集群源码有过未 push 的本地修改，
> 上面的 `git diff` 会先列出来，必要时自行备份。之后集群编译 `.so` 照旧：
> `MAX_JOBS=4 python3 setup.py build_ext --inplace`。

## 日常流程（每次跑验证）

```bash
cd /hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa
git pull                      # 代码 + tests/ + 本工具一次到位
sbatch scripts/cluster/job_validate_all.sh   # 日志 ./log_validate_all.out
```

真实分析对照（可选）：

```bash
export CONFIG=/path/to/analysis/config.yml
sbatch scripts/cluster/job_validate_all.sh
```

日志回传本地（本机执行，走集群登录节点）：

```bash
bash scripts/cluster/sync_to_cluster.sh pull   # 拉到 scripts/cluster/logs/
# 或手动: scp 集群:.../log_validate_all.out .
```

## 不再需要的手工同步

旧流程（tests_cluster/ 散文件 scp 到集群 tests/）已废弃：tests/ 已入库，集群是 git 仓库后
`git pull` 取代一切文件搬运。历史归档见仓库内 `tests_cluster/_archive_old_debug/` 与本地
`~/pwa/debug/tests_archive_*/`。
