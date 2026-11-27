#ifndef AMPLITUDE_CUH
#define AMPLITUDE_CUH

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// 四矢量结构体（__device__ __host__ 双用, 公式与 ROOT TLorentzVector/TVector3 逐条对齐）
struct LorentzVector {
    double E, Px, Py, Pz;

    __device__ __host__ LorentzVector() : E(0), Px(0), Py(0), Pz(0) {}
    __device__ __host__ LorentzVector(double e, double px, double py, double pz)
        : E(e), Px(px), Py(py), Pz(pz)
    {
    }

    __device__ __host__ double P() const
    {
        return sqrt(Px * Px + Py * Py + Pz * Pz);
    }

    __device__ __host__ double M() const
    {
        if (E * E - Px * Px - Py * Py - Pz * Pz < 0)
            return 0.0;
        return sqrt(E * E - Px * Px - Py * Py - Pz * Pz);
    }

    __device__ __host__ double M2() const
    {
        return E * E - Px * Px - Py * Py - Pz * Pz;
    }

    __device__ __host__ double Dot(const LorentzVector &other) const
    {
        return E * other.E - (Px * other.Px + Py * other.Py + Pz * other.Pz);
    }

    __device__ __host__ LorentzVector
    operator+(const LorentzVector &other) const
    {
        return LorentzVector(E + other.E, Px + other.Px, Py + other.Py,
                             Pz + other.Pz);
    }

    // ---- 以下与 ROOT TLorentzVector / TVector3 公式逐条对齐（注释标注 ROOT 源引） ----

    // TVector3::Perp() = sqrt(px²+py²)
    __device__ __host__ double Perp() const
    {
        return sqrt(Px * Px + Py * Py);
    }

    // TVector3::Rho() 及 TLorentzVector::Pt() 同义
    __device__ __host__ double Pt() const { return Perp(); }

    // TVector3::Theta() = acos(pz/|p|)（|p|=0 时返回 0; ROOT 原版为 NaN, 这里取安全约定）
    __device__ __host__ double Theta() const
    {
        double p = P();
        return (p == 0.0) ? 0.0 : acos(Pz / p);
    }

    // TVector3::CosTheta() = pz/|p|（|p|=0 时返回 1 的安全约定）
    __device__ __host__ double CosTheta() const
    {
        double p = P();
        return (p == 0.0) ? 1.0 : Pz / p;
    }

    // TVector3::Phi() = atan2(py, px) ∈ (-π, π]（px=py=0 时返回 0）
    __device__ __host__ double Phi() const
    {
        return (Px == 0.0 && Py == 0.0) ? 0.0 : atan2(Py, Px);
    }

    // TLorentzVector::Angle(v): 三维动量夹角, cos 夹紧到 [-1,1]
    __device__ __host__ double Angle(const LorentzVector &q) const
    {
        double p = P(), qq = q.P();
        if (p == 0.0 || qq == 0.0) return 0.0;
        double c = (Px * q.Px + Py * q.Py + Pz * q.Pz) / (p * qq);
        c = (c < -1.0) ? -1.0 : ((c > 1.0) ? 1.0 : c);
        return acos(c);
    }

    // TLorentzVector::Boost(b): 与 ROOT 同款公式与符号约定
    //   E' = γ(E + β·p),  p' = p + (γ-1)/β² (β·p)β + γ E β   （被动约定: 新系以 -b 运动）
    __device__ __host__ void Boost(double bx, double by, double bz)
    {
        double b2 = bx * bx + by * by + bz * bz;
        if (b2 == 0.0) return;
        double bp = bx * Px + by * Py + bz * Pz;
        double gamma = 1.0 / sqrt((b2 < 1.0) ? (1.0 - b2) : 1e-30);
        double coeff = (gamma - 1.0) / b2;
        double px = Px + coeff * bp * bx + gamma * bx * E;
        double py = Py + coeff * bp * by + gamma * by * E;
        double pz = Pz + coeff * bp * bz + gamma * bz * E;
        double e = gamma * (E + bp);
        Px = px; Py = py; Pz = pz; E = e;
    }

    // TLorentzVector::BoostToRest(p): 变换到 p 的静系（= Boost(-p⃗/E)）
    __device__ __host__ void BoostToRest(const LorentzVector &mother)
    {
        Boost(-mother.Px / mother.E, -mother.Py / mother.E,
              -mother.Pz / mother.E);
    }

    // 便于 boost 后取 3-矢量夹角余弦（= cos(Angle(q))）
    __device__ __host__ double CosAngle(const LorentzVector &q) const
    {
        double p = P(), qq = q.P();
        if (p == 0.0 || qq == 0.0) return 1.0;
        double c = (Px * q.Px + Py * q.Py + Pz * q.Pz) / (p * qq);
        return (c < -1.0) ? -1.0 : ((c > 1.0) ? 1.0 : c);
    }
};

__device__ void pwa_amp(thrust::complex<double> *amp, LorentzVector p1,
                        int dim_j1, LorentzVector p2, int dim_j2, int dim_j,
                        int dim_S, int dL, thrust::complex<double> *shared_buf);

#endif // AMPLITUDE_CUH
