# 大改动流程：改 → 测 → 发（不干扰现有框架）

> 问题：能不能"先建个新文件夹、在那边改，测过再发布"，同时把 push GitHub → 集群验证 →
> 发布的完整流程建立起来？
>
> **结论：不要用"新文件夹并行开发"，用"长期分支 + 冻结版本做 oracle"。**
> 新文件夹唯一的合理用途是 A/B 对照，而那个用**已发布的 0.3.26 wheel** 更便宜、
> 更可靠、零维护。开发本身留在原地，靠分支隔离。
>
> 盘点基准 `56d500e`（v0.3.26）。

---

## 0. 一句话

| 想法 | 判断 |
|---|---|
| "新文件夹里重写，测完再切" | ❌ 本仓库代价高（见 §1），且切换 = 30k 行一次性 merge |
| "新文件夹当对照基线" | ✅ 但用**冻结 wheel** 更好：冻结、可复现、不占仓库 |
| "分支 + 分级门禁 + rc 先发" | ✅ 推荐（§2–§6） |
| "改完只跑对应测试" | ⚠️ 只适合开发期快查；门禁必须全量（见 modularization-plan §7.4） |

---

## 1. 为什么"并行目录"在本仓库代价高

四条都是这个仓库的具体事实，不是一般性教条：

1. **`.gitignore` 是 `*` + 白名单**。新顶层目录默认**不入库**，必须显式加
   `!src2/` + `!src2/**`。漏了就是"本地能跑、推上去没了"；`git add -f` 又会让
   ignore 语义更难维护。
2. **一个进程只能加载一个 `.so`**（pybind11 类型全局注册；旧新同名类型会撞
   `type ... is already registered`）。所以"两个目录两套产物"并不能让新旧同进程
   A/B，必须靠 `setup.py` 里加源码集开关 —— 那是**永久的**构建复杂度，只是为了
   一次重构。
3. **上游还在动**：近 60 次提交里 `AmpGen.cu` 改了 24 次、`Analysis.cu` 15 次。
   双树期间每个修复要 port 两遍，漂移是必然的，不是可能的。
4. **切换是一次 30k 行 merge**。重构的全部风险恰好集中在那一天，而不是分散在
   每个小 PR —— 与"降低风险"的目标相反。

对比：分支 + 冻结 oracle 的成本是一个 `git switch`，且"旧版本"是**同一个二进制**，
不会随开发漂移。

---

## 2. 分支模型

```
main                     只有已验证内容；tag 从这里打
refactor/modular         长期分支，P0 → P4 顺序合并
feat/p0-device-guard     每阶段一个短分支 → PR 到 refactor/modular
```

规则（4 条，都是硬约束）：

| 规则 | 理由 |
|---|---|
| `refactor/modular` **任何时候都全绿** | "永远可发布"：任何一天叫停，剩下的分支都能单独发 |
| 每周至少 rebase 一次 `main` | 冲突以天为单位暴露，而不是以月 |
| 单阶段 ≤ 5 天；**>2 周不绿就拆小或停** | 每个阶段独立可发布，叫停只需不合并下一个 PR |
| 每个 PR 单一主题 | P0 的 219 处机械替换单独成 PR，review 靠 `grep` 证明"无残留"而不是逐行读 |

---

## 3. 分级门禁

| 级别 | 在哪 | 跑什么 | 时长 | 何时 |
|---|---|---|---|---|
| **G1 快查** | 本机 | `CTPWA_BACKEND=cpu bash scripts/run_tests.sh --no-build` | 3–8 min | 每次改完 |
| **G2 全量** | 本机 | `bash scripts/run_tests.sh`（含编译，CPU 后端） | 20–40 min | 提交前 |
| **G3 CI** | GitHub runner | CPU 构建 + `pytest tests/`（含 golden） | 30–60 min | PR / push main |
| **G4 单卡** | V100×1 | 全量 + 真实数据 L2 + `job_validate_all.sh` | 1–2 h | 每个 PR 合并前 |
| **G5 双卡** | V100×2 | `multigpu_parity.py` + 显存预检 | 30–60 min | 每个阶段结束 |
| **G6 rc** | PyPI pre-release | `pip install --pre ctpwa==0.4.0rcN` 跑真实分析 | — | 正式发布前 |

**G6 是"不影响现有用户"的关键机制**：`0.4.0rc1` 在 PyPI 上是 pre-release，
`pip install ctpwa` 默认拿不到它，只有显式 `--pre` 才能装。所以"发布候选"可以直接
走真实的安装路径验证，而不触碰 stable。

**G3 今天是缺的**：`.github/workflows/` 只有 tag 触发的 `build-wheel.yml`，
push/PR 没有任何自动检查 —— "测完再发"完全依赖人记性。
补法见 §9；注意它有个前置：**先让测试在无卡机器上能跑绿**（§9.0）。

---

## 4. A/B oracle：冻结版本对照

不建第二棵树，建**第二个 venv**：

| venv | 内容 | 用途 |
|---|---|---|
| A（冻结） | `pip install ctpwa==0.3.26`，或离线 wheel 放 AFS `~/wheels/` | 基线 |
| B（当前） | 分支 build 的 `.so`，`PYTHONPATH=<checkout>` | 被测 |

`scripts/parity/ab_compare.py <config.yml> --seed 42`：

1. 起两个**子进程**（不能同进程，见 §1.2），各自 dump
   `{nll, grad, hess, fractions, interference, matrix_err}` 到 json；
2. 逐值比较，容差沿用现有后端判据：NLL 1e-11 / 梯度 1e-10 / Hessian 1e-12（相对）；
3. 结果写 `doc/parity/phase-<n>.json` 入库。

**P0 的判据更严：必须逐位相等（容差 0）。** P0 是纯机械替换，任何数值差都说明
"替换引入了语义变化"，直接回退——这比"在容差内"有信息量得多。后续阶段
（P2/P3 动所有权与类型）才退回容差判据。

---

## 5. 集群：目录隔离

现有 `scripts/cluster/job_validate_all.sh` 里 `CTPWA_ROOT` 是**硬编码**的：

```bash
CTPWA_ROOT=/hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa     # 生产 checkout
```

规则：

1. **不要把分支放进这个目录**——那里可能在跑生产作业。
2. 每分支独立目录：`.../pwatest/ctpwa-refactor`，`git clone -b refactor/modular`。
3. 把脚本改成可覆盖（现在不是 `:-` 默认，是写死）：
   `CTPWA_ROOT="${CTPWA_ROOT:-/hpcfs/.../pwatest/ctpwa}"`。
4. 成本很低：`build/` 70 MB + `_cuda.so` 15 MB + `_cpu.so` 3.5 MB。
5. 作业矩阵：单卡跑全量；双卡跑 parity；真实 config 用现成的 `CONFIG=` 支持。
6. 每个作业末尾追加写 `validate.json`（pass/fail + 关键行），PR 里贴这个文件，
   **不要把 `log_*.out` 全文塞进 PR**。

---

## 6. 版本与发布

| 步骤 | 命令 | 说明 |
|---|---|---|
| 抬版本 | `setup.py` → `0.4.0rc1` | CI 要求 **tag 名 == version**，rc 也一视同仁 |
| 打 rc | `git tag v0.4.0rc1 && git push origin v0.4.0rc1` | 触发 wheel 构建，PyPI 上是 pre-release |
| 集群验证 rc | `pip install --pre ctpwa==0.4.0rc1` | 走真实安装路径，不动 stable |
| 正式 | 合 `main` → `0.4.0` → `git tag v0.4.0` | stable |

`CHANGELOG.md` 的 `[Unreleased]` 段在**每个 PR** 里更新（仓库已有此约定），
发布时整段改名为 `## [v0.4.0]`。

---

## 7. "不动现有框架"的可执行定义

"不动"不是"不改文件"，是**对外契约冻结**。这张表就是重构期间的红线：

| 冻结项 | 判据 |
|---|---|
| `ctpwa.analysis(...)` 构造 | 只允许加**带默认值**的关键字参数 |
| 公开方法名 / 返回 dtype / shape | `getNLL` `getHessian` `getFitFractions` `writeResult` `getPhspTensor` … 全部不变 |
| `.so` 文件名与后端门面 | `ctpwa/_cuda.so`、`_cpu.so`；`ctpwa.backend()` / `build_info()` 语义 |
| `config.yml` 语义 | 现有键默认值不变；新增键必须可选（老配置照样跑） |
| `example/fit.py`、`scripts/cluster/*.py` | 不改一行仍可用（它们是外部使用者的代表） |

**明确允许改**（重构自由度）：文件位置、内部类名、私有方法、kernel 实现、
`src/` 目录结构、`main.cu` 的包含方式。

判据一句话：**外部脚本一行不改还能跑，内部怎么动都行。**

---

## 8. 每阶段 PR checklist

```markdown
- [ ] G1/G2 绿（本机）
- [ ] G3 CI 绿
- [ ] G4 单卡绿 —— 附 log 名
- [ ] G5 双卡绿（涉及 device 的 PR 必填）
- [ ] A/B parity json 入库；P0 差值为 0，其余在容差内
- [ ] CHANGELOG [Unreleased] 已更新
- [ ] §7 红线逐项确认无变更
- [ ] 回滚说明：revert 本 PR 即回到上一个绿点
```

---

## 9. 第一步（零风险，不碰源码）

### 9.0 先修一个拦路的东西：无卡机器上全量跑不起来

**实测**（CPU 构建 + 无 GPU，`pytest tests/test_custom.py`）：

```
ERROR test_custom.py::test_custom_equals_bw - AssertionError: ctpwa 需要 CUDA GPU
```

原因在 `tests/conftest.py:96` 的 `device` fixture：

```python
@pytest.fixture(scope="session")
def device():
    assert torch.cuda.is_available(), "ctpwa 需要 CUDA GPU"   # ← assert 而不是 skip
```

用到 `device` 的 6 个文件（`test_custom` / `test_custom_models` / `test_custom_q0` /
`test_float_mode` / `test_precision` / `test_static_budget`）在 CPU 构建上会 **error**
而不是 **skip**。也就是说：**今天没有任何"无卡机器能跑绿"的全量门禁**，
这是 G3 缺失的根因，不只是 yml 没写。

修法一行：

```python
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA GPU（当前无可用设备）")
```

> 语义变化要说清楚：从"断言环境不对"变成"跳过"。这会让本地忘记开 GPU 时**静默变绿**，
> 所以配套打印一行 `skipped N (no cuda)` 汇总，别让跳过变成隐形。

**前置 2：先拿到一次干净基线。** 我用仓库里现成的 `_cpu.so`（`17:01`，比
`src/` 的 `17:09` 旧 8 分钟）跑全量，到 ~50% 时既有 `E` 也有 `F`，还有一个用例
长时间不动。`E` 是上面那条 fixture 的确定性后果；`F` 与挂起**不能作数**——二进制
与源码不同源。所以做 CI 之前必须先：

```bash
CTPWA_BACKEND=cpu MAX_JOBS=4 python3 setup.py build_ext --inplace   # 干净重建
bash scripts/run_tests.sh --no-build
```

把这一轮的数字（passed / failed / skipped / 耗时）记进 `doc/`，它才是 G2 的基线。
没有基线就上 CI，第一次红你会分不清是重构引入的还是原本就红。

### 9.1 然后加 CI

新增 `.github/workflows/ci.yml`，内容几乎可以照抄 `build-wheel.yml` 的第 2 步：

```yaml
# 关键点（其余照抄 build-wheel.yml）
- name: conda env
  run: |
    # conda-forge 的 root 提供 root-config —— 这个项目唯一可靠的 ROOT 来源
    # （build-wheel.yml 用 miniconda + --override-channels -c conda-forge，
    #   因为 defaults 源与 conda-forge 的 root 依赖链 solve 冲突）
    conda create -y -q -p /opt/env --override-channels -c conda-forge \
      python=3.12 root ninja
- name: torch (CPU)
  run: pip install torch --index-url https://download.pytorch.org/whl/cpu
- name: build (CPU 后端)
  run: CTPWA_BACKEND=cpu MAX_JOBS=4 python3 setup.py build_ext --inplace
- name: gate
  run: bash scripts/run_tests.sh --no-build
```

成本与取舍：

- conda 环境（ROOT 依赖链）+ CPU 全量编译 ≈ **30–60 min**（视 runner 核数），所以只在
  **PR 与 push main** 触发，feature 分支只 build 不测；配 conda/pip 缓存。
- 无卡 runner 上约 6 个文件、~30 个用例会 skip（§9.0 之后），其余全跑；
  GPU 相关能力仍只能靠 G4/G5 覆盖。
- 这一步的全部意义：**先有门禁，再动代码。** 没有它，后面每个阶段的验收都靠
  人记性。

---

## 10. 与 `modularization-plan.md` 的对应

| 阶段 | 门禁要求 | A/B 判据 |
|---|---|---|
| P0 设备纪律 | G1–G5 全绿 | 逐位相等（0 容差） |
| P1 真 TU | G1–G4 | 逐位相等 |
| P2 DevBuffer | G1–G5 | 容差内 |
| P3 DecayInfo 拆分 | G1–G4 + 新增 `unit/` 用例 | 容差内 |
| P4 god class | G1–G5 + `pipeline/` 用例 | 容差内 |

P0/P1 的"逐位相等"是这套流程里最强的护栏：它们看起来改动面最大（219 处 +
12.8k 行改 TU），但**不改任何语义**，所以可以用 0 容差卡死。

---

## 11. 远端迁移到 GitLab

**可行，且很便宜**：`.git` 25 MB，最大对象 2 MB（`tests/data/*_phsp.dat`），
不需要 LFS，远低于任何 push 限额。现有 11 个本地分支 / 27 个 release tag。

`git remote add` 不动任何历史，所以这一步天然可回滚。

### 三档，选哪档决定代价

| 档 | 做法 | 代价 | 适用 |
|---|---|---|---|
| **A 备份** | 加一个 remote + push | 极小 | 只想有内网副本 |
| **B 主仓 + 发布镜像**（推荐） | 开发/MR/评审在 GitLab；GitLab 的 **push mirror** 把 `main` + tags 同步到 GitHub；GitHub Actions 发布链路原样保留 | 中（配 mirror、确认 tag 同步） | 内网速度、集群访问 |
| **C 全搬** | 重写 `.gitlab-ci.yml`、迁移 `PYPI_API_TOKEN`、改 README/PyPI 元数据 | 大 | 彻底不用 GitHub |

推荐 B 的两条硬理由（都是本仓库现状）：

1. **能顺手干掉 AFS 裸库这个 hack**。现在 `cluster` 远程指向
   `bhxiang@lxlogin.ihep.ac.cn:/afs/.../ctpwa.git`，唯一用途是让集群 `git pull`。
   GitLab 在内网，可以**直接当这个角色** → remote 从"GitHub + AFS"变成
   "GitLab + GitHub 镜像"，反而更少。
2. **发布链路不用重写**。PyPI 发布是 `pypa/gh-action-pypi-publish`（Docker action）
   + `secrets.PYPI_API_TOKEN`，跑在 GitHub 上。mirror 过去后这套原样工作。
   改成 GitLab CI 是这次迁移里**唯一真正贵**的部分：需要能跑 `almalinux:9` 容器的
   runner（CUDA toolkit 12.9 + conda ROOT，几十 GB 磁盘）、token 重配为 protected
   variable、`tag==version` 检查重写。

### 命令（A 档，10 秒）

```bash
git remote add gitlab git@gitlab.ihep.ac.cn:<group>/ctpwa.git
git push gitlab --all      # 11 个分支（含 backup-before-perSL-fix 这类，可先清理）
git push gitlab --tags     # 27 个 release tag，必须带 —— 否则 release 断层
```

### 本仓库特有的坑

| 坑 | 说明 |
|---|---|
| `.gitignore` 是 `*` 白名单 | 换到哪个远端都一样：`doc/` 目录 **0 个文件在库内**。迁移前先决定 doc 放不放行，否则"搬完发现文档没搬" |
| tag 必须一起推 | 不推 tag → 已发布版本的来源不可追溯；PyPI 上的 `pip install ctpwa==0.3.26` 靠 tag 对应 |
| 别双写 main | 同时 `git push` 两个 remote + 一次 rebase 就会分叉。用 mirror **单向**同步 |
| 内网可见性 | IHEP GitLab 多为内网/私有 → PyPI 页面与 README 的 `git clone https://github.com/BHXiang/CTPWA.git` 仍应指向 GitHub |
| 密钥 | `PYPI_API_TOKEN` 只放 protected variable，且只对 protected tag 生效；任何仓库文件里都不能出现 |

### 对重构流程的影响

分支模型、6 级门禁、A/B 判据全部不变。变的只有三处：

- `git push origin` → `git push gitlab`（或改 `origin` 指向）；
- `scripts/cluster/job_validate_all.sh` 的 `CTPWA_ROOT` 指向从 GitLab clone 的目录
  （顺便改成 `${CTPWA_ROOT:-...}`）；
- MR 评审换到 GitLab Merge Request，`main` 开 **protected branch**（自建实例比
  GitHub 更好配），效果与 branch protection 相同。
