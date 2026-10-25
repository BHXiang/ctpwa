#!/usr/bin/env python3
"""port_cuda_to_cpu.py — 把 CUDA 启动语法批量改成 CTPWA_LAUNCH（一次性迁移工具）

只做三件机械替换：
  1. `K<<<g, b>>>(args)` / `K << <g, b >> > (args)`  →  `CTPWA_LAUNCH(K, g, b, 0, args)`
     带第三个参数（动态 shared 字节数）时原样保留。
  2. `__global__ void` → `CTPWA_KERNEL void`
  3. `int x = blockIdx.x * blockDim.x + threadIdx.x;` → `const int x = CTPWA_TID();`

二维索引（blockIdx.y / threadIdx.y）**不处理**——这类 kernel 必须按 CPU 形态重写，
留给人工（见 doc/cpu-backend-plan.md §P1/P3）。

用法: python3 scripts/port_cuda_to_cpu.py src/Foo.cu [...]
"""
import re
import sys

KERNEL_HEAD = re.compile(r'__global__(\s+)void')
TID_PATTERNS = [
    re.compile(r'int\s+(\w+)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x\s*;'),
    re.compile(r'unsigned\s+int\s+(\w+)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x\s*;'),
]


def _find_close(src, start):
    """找到 `<<<` 之后**最近**的配置结束标记，返回 (end_index, marker)。

    注意必须取三者中最近的一个：早期版本按 `>>>` → `>> >` 顺序找，
    遇到 `< < ... >> >` 写法会一路找到文件后面某个 `>>>`，把整段吞掉。
    """
    best = None
    for marker in ('>>>', '>> >', '> > >'):
        k = src.find(marker, start)
        if k >= 0 and (best is None or k < best[0]):
            best = (k, marker)
    return best if best else (-1, None)


def _split_top(cfg):
    parts, depth, cur = [], 0, ''
    for ch in cfg:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def rewrite_launches(src):
    out, i, n_done = [], 0, 0
    while True:
        j1 = src.find('<<<', i)
        j2 = src.find('<< <', i)
        cands = [x for x in (j1, j2) if x >= 0]
        if not cands:
            out.append(src[i:])
            break
        j = min(cands)
        end, marker = _find_close(src, j + 1)
        if end < 0:
            out.append(src[j:])
            break
        cfg = src[j + len('<<<') if src.startswith('<<<', j) else j + len('<< <'):end]
        parts = _split_top(cfg)
        if len(parts) not in (2, 3):
            # 不是标准启动配置，原样保留
            out.append(src[i:end + len(marker)])
            i = end + len(marker)
            continue
        grid, block = parts[0], parts[1]
        shm = parts[2] if len(parts) == 3 else '0'

        # kernel 名：紧邻 <<< 之前（可能带模板实参）。直接在 src 上回看，
        # 不依赖已累积的 out，避免重复插入。
        m = re.search(r'([A-Za-z_]\w*(?:\s*<[^<>]*>)?)\s*$', src[:j])
        if not m:
            out.append(src[i:end + len(marker)])
            i = end + len(marker)
            continue
        name = m.group(1)
        out.append(src[i:m.start()])

        # 配置之后应当是 '('
        after = end + len(marker)
        k = after
        while k < len(src) and src[k] in ' \t\r\n':
            k += 1
        if k >= len(src) or src[k] != '(':
            out.append(src[i:after])
            i = after
            continue

        out.append('CTPWA_LAUNCH(%s, %s, %s, %s, ' % (name, grid, block, shm))
        i = k + 1      # 跳过 '('，内容原样保留，末尾 ')' 关闭宏调用
        n_done += 1
    return ''.join(out), n_done


def main(paths):
    for p in paths:
        s = open(p, encoding='utf-8').read()
        orig = s
        s, n = rewrite_launches(s)
        s, nk = KERNEL_HEAD.subn(r'CTPWA_KERNEL\1void', s)
        nt = 0
        for pat in TID_PATTERNS:
            s, k = pat.subn(r'const int \1 = CTPWA_TID();', s)
            nt += k
        if s != orig:
            open(p, 'w', encoding='utf-8').write(s)
        print('%-28s launches=%d kernels=%d tid=%d' % (p, n, nk, nt))


if __name__ == '__main__':
    main(sys.argv[1:])
