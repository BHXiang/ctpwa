#ifndef AMPLITUDE_CUH
#define AMPLITUDE_CUH

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// 四矢量结构体
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
};

__device__ void pwa_amp(thrust::complex<double> *amp, LorentzVector p1,
                        int dim_j1, LorentzVector p2, int dim_j2, int dim_j,
                        int dim_S, int dL, thrust::complex<double> *shared_buf);

#endif // AMPLITUDE_CUH
