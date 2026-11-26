// test_decay_desc.cpp — 衰变描述类型的纯 host 单测（不需要 GPU / torch / ROOT）
//
// P3 的验收：`DecayChainDesc` 能在没有任何设备、没有任何 config 文件的情况下被
// 构造与检查。这里只 include 两个头 + 链 DecayDesc.cu，链接依赖只有标准库。
#include <DecayChainDesc.h>
#include <DecayDesc.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);  \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// J/psi(1-−) → eta(0-+) + gamma(1-−)? 用更简单的一组：1- → 0-+ + 1--
static Amp2BD make_step()
{
    return Amp2BD(std::array<int, 3>{3, 1, 3},      // 2J+1：J=1, J=0, J=1
                  std::array<int, 3>{-1, -1, -1},   // 宇称
                  /*identical_daughters=*/false, /*is_boson=*/true,
                  /*maxL=*/4);
}

static void test_amp2bd_sl()
{
    Amp2BD amp = make_step();
    CHECK(!amp.getSL().empty());
    // 结构不变量（不依赖宇称约定）：
    //   两体耦合满足 |J2-J3| <= S <= J2+J3（这里存的是 2S+1），且 L 不超过 maxL。
    //   jvalues 存 2J+1：J = (v-1)/2。
    constexpr int maxL = 4;
    for (const auto& sl : amp.getSL()) {
        const int S = sl.S - 1;            // 物理自旋 S = (2S+1-1)
        CHECK(S >= 0);
        CHECK(S <= 2);                     // 两个 J=1 的末态，S 最多 2
        CHECK(sl.L >= 0);
        CHECK(sl.L <= maxL);
    }
}

static void test_desc_dedup_and_order()
{
    ctpwa::DecayChainDesc d;
    d.addParticle("Jpsi", 3, -1, 3.0969);
    d.addParticle("eta", 1, -1, 0.5479);
    d.addParticle("Jpsi", 3, -1, 3.0969);   // 重复登记应被忽略

    CHECK(d.particles().size() == 2);
    CHECK(d.particleNames().size() == 2);
    CHECK(d.particleNames()[0] == "Jpsi");   // 保持首次出现顺序
    CHECK(d.particles().at("eta").mass > 0.5);
}

static void test_desc_nodes_and_waves()
{
    ctpwa::DecayChainDesc d;
    Amp2BD a = make_step();
    Amp2BD b = make_step();
    d.addNode(DecayNodeHost{a, "Jpsi", "eta", "gamma"}, a.getSL().size());
    CHECK(d.size() == 1);
    const std::size_t after_one = d.nSLCombs();
    CHECK(after_one == a.getSL().size());

    d.addNode(DecayNodeHost{b, "gamma", "e+", "e-"}, b.getSL().size());
    CHECK(d.size() == 2);
    CHECK(d.nSLCombs() == after_one * b.getSL().size());

    // 表侧可能按过滤后的实际波数覆盖
    d.setNSLCombs(7);
    CHECK(d.nSLCombs() == 7);

    // 节点字段原样保留
    CHECK(d.nodes()[0].mother == "Jpsi");
    CHECK(d.nodes()[1].daug2 == "e-");
}

static void test_desc_empty()
{
    ctpwa::DecayChainDesc d;
    CHECK(d.empty());
    CHECK(d.nSLCombs() == 1);   // 空链的乘积约定为 1
}

int main()
{
    std::printf("[test_decay_desc] 纯 host，无 GPU/torch/ROOT 依赖\n");
    test_amp2bd_sl();
    test_desc_dedup_and_order();
    test_desc_nodes_and_waves();
    test_desc_empty();
    if (g_fail) {
        std::printf("[test_decay_desc] FAILED: %d\n", g_fail);
        return EXIT_FAILURE;
    }
    std::printf("[test_decay_desc] all checks passed\n");
    return EXIT_SUCCESS;
}
