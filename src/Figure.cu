#include <Figure.cuh>
// #include <Amplitude.cuh>

// CUDA核函数：填充直方图（有权重）
__global__ void fillHistogramKernel(const double *values, const double *weights,
                                    int n_events, double *hist_bins, int n_bins,
                                    double min_bin, double max_bin,
                                    double bin_width)
{
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (event_idx < n_events) {
        double value = values[event_idx];
        double weight = 1.0;
        if (weights != nullptr)
            weight = weights[event_idx];

        if (value >= min_bin && value < max_bin) {
            int bin_idx = static_cast<int>((value - min_bin) / bin_width);
            if (bin_idx < 0)
                bin_idx = 0;
            if (bin_idx >= n_bins)
                bin_idx = n_bins - 1;

            // 使用原子操作确保线程安全
            atomicAdd(&hist_bins[bin_idx], weight);
        }
    }
}

// CUDA核函数：计算每个事件的质量
__global__ void MassCalculator(const LorentzVector *momenta,
                               const int *particle_indices,
                               int n_particles_in_config, int total_particles,
                               int n_events, double *masses)
{
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (event_idx < n_events) {
        LorentzVector total;
        for (int i = 0; i < n_particles_in_config; ++i) {
            int particle_idx = particle_indices[i];
            const LorentzVector &p =
                momenta[event_idx * total_particles + particle_idx];
            total = total + p;
        }
        masses[event_idx] = total.M();
    }
}

void CalculateMassHist(LorentzVector *device_momenta,
                       const std::map<std::string, int> &particleToIndex,
                       const std::vector<MassHistConfig> &histConfigs,
                       double *device_weights,
                       std::vector<TH1F *> &outputHistograms, int nEvents,
                       int nParticles)
{
    if (outputHistograms.size() != histConfigs.size()) {
        std::cerr << "Error: outputHistograms size (" << outputHistograms.size()
                  << ") does not match histConfigs size (" << histConfigs.size()
                  << ")" << std::endl;
        return;
    }

    if (nEvents == 0) {
        std::cerr << "Warning: No events to process!" << std::endl;
        return;
    }

    // CUDA配置
    int blockSize = 256;
    int gridSize = (nEvents + blockSize - 1) / blockSize;

    for (size_t configIdx = 0; configIdx < histConfigs.size(); ++configIdx) {
        const auto &config = histConfigs[configIdx];
        TH1F *hist = outputHistograms[configIdx];

        if (!hist) {
            std::cerr << "Error: outputHistograms[" << configIdx << "] is null!"
                      << std::endl;
            continue;
        }

        hist->Reset();

        // 获取粒子索引
        std::vector<int> particleIndices;
        for (const auto &particleName : config.particles) {
            auto it = particleToIndex.find(particleName);
            if (it == particleToIndex.end()) {
                std::cerr << "Error: Particle '" << particleName
                          << "' not found in particleToIndex map!" << std::endl;
                particleIndices.clear();
                break;
            }
            particleIndices.push_back(it->second);
        }

        if (particleIndices.empty()) {
            std::cerr << "Warning: Config " << configIdx
                      << " has no valid particles!" << std::endl;
            continue;
        }

        if (config.range.size() < 2) {
            std::cerr << "Error: Config " << configIdx << " range size < 2!"
                      << std::endl;
            continue;
        }

        double min_bin = config.range[0];
        double max_bin = config.range[1];
        int n_bins = config.bins;
        double bin_width = (max_bin - min_bin) / n_bins;

        // 在设备上分配粒子索引数组
        int *device_particle_indices;
        cudaMalloc(&device_particle_indices,
                   particleIndices.size() * sizeof(int));
        cudaMemcpy(device_particle_indices, particleIndices.data(),
                   particleIndices.size() * sizeof(int),
                   cudaMemcpyHostToDevice);

        // 步骤1：计算所有事件的质量
        double *device_masses;
        cudaMalloc(&device_masses, nEvents * sizeof(double));

        MassCalculator<<<gridSize, blockSize>>>(
            device_momenta, device_particle_indices, particleIndices.size(),
            nParticles, nEvents, device_masses);
        cudaDeviceSynchronize();

        // 步骤2：填充直方图
        double *device_hist_bins;
        cudaMalloc(&device_hist_bins, n_bins * sizeof(double));
        cudaMemset(device_hist_bins, 0, n_bins * sizeof(double));

        fillHistogramKernel<<<gridSize, blockSize>>>(
            device_masses, device_weights, nEvents, device_hist_bins, n_bins,
            min_bin, max_bin, bin_width);
        cudaDeviceSynchronize();

        // 步骤3：将直方图结果复制回主机
        std::vector<double> host_bin_counts(n_bins, 0.0);
        cudaMemcpy(host_bin_counts.data(), device_hist_bins,
                   n_bins * sizeof(double), cudaMemcpyDeviceToHost);

        // 填充TH1F直方图
        for (int bin = 0; bin < n_bins; ++bin) {
            hist->SetBinContent(bin + 1, host_bin_counts[bin]);
        }
        hist->SetBins(n_bins, min_bin, max_bin);

        // 清理设备内存
        cudaFree(device_particle_indices);
        cudaFree(device_masses);
        cudaFree(device_hist_bins);

        // std::cout << "Processed config " << configIdx << " with " << nEvents
        // << " events" << std::endl;
    }
}

// 角度计算核函数
__global__ void AngleCalculator(const LorentzVector *momenta,
                                const int *particle_indices,
                                int *n_particles_in_config, int total_particles,
                                int n_events, double *angles)
{
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (event_idx < n_events) {
        int idx_offset = 0;

        LorentzVector p;
        for (int i = 0; i < n_particles_in_config[0]; ++i) {
            int particle_idx = particle_indices[idx_offset + i];
            p = p + momenta[event_idx * total_particles + particle_idx];
        }
        idx_offset += n_particles_in_config[0];

        LorentzVector q;
        for (int i = 0; i < n_particles_in_config[1]; ++i) {
            int particle_idx = particle_indices[idx_offset + i];
            q = q + momenta[event_idx * total_particles + particle_idx];
        }
        idx_offset += n_particles_in_config[1];

        LorentzVector d;
        for (int i = 0; i < n_particles_in_config[2]; ++i) {
            int particle_idx = particle_indices[idx_offset + i];
            d = d + momenta[event_idx * total_particles + particle_idx];
        }

        double pd = p.Dot(d);
        double qd = q.Dot(d);
        double pq = p.Dot(q);
        double mp2 = p.M2();
        double mq2 = q.M2();
        double md2 = d.M2();

        double denominator = (pq * pq - mq2 * mp2) * (qd * qd - mq2 * md2);
        if (denominator <= 0) {
            angles[event_idx] = 0.0;
            return;
        }
        double cost = (pd * mq2 - pq * qd) / sqrt(denominator);

        angles[event_idx] = cost;
    }
}

void CalculateAngleHist(LorentzVector *device_momenta,
                        const std::map<std::string, int> &particleToIndex,
                        const std::vector<AngleHistConfig> &histConfigs,
                        double *device_weights,
                        std::vector<TH1F *> &outputHistograms, int nEvents,
                        int nParticles)
{
    if (outputHistograms.size() != histConfigs.size()) {
        std::cerr << "Error: outputHistograms size (" << outputHistograms.size()
                  << ") does not match histConfigs size (" << histConfigs.size()
                  << ")" << std::endl;
        return;
    }

    if (nEvents == 0) {
        std::cerr << "Warning: No events to process!" << std::endl;
        return;
    }

    // CUDA配置
    int blockSize = 256;
    int gridSize = (nEvents + blockSize - 1) / blockSize;

    for (size_t configIdx = 0; configIdx < histConfigs.size(); ++configIdx) {
        const auto &config = histConfigs[configIdx];
        TH1F *hist = outputHistograms[configIdx];

        if (!hist) {
            std::cerr << "Error: outputHistograms[" << configIdx << "] is null!"
                      << std::endl;
            continue;
        }

        hist->Reset();

        // 获取粒子索引
        std::vector<int> particleIndices;
        std::vector<int> groupSizes;

        for (const auto &particleGroup : config.particles) {
            groupSizes.push_back(particleGroup.size());
            for (const auto &particleName : particleGroup) {
                auto it = particleToIndex.find(particleName);
                if (it == particleToIndex.end()) {
                    std::cerr << "Error: Particle '" << particleName
                              << "' not found in particleToIndex map!"
                              << std::endl;
                    particleIndices.clear();
                    groupSizes.clear();
                    break;
                }
                particleIndices.push_back(it->second);
            }
            if (particleIndices.empty())
                break;
        }

        if (particleIndices.empty()) {
            std::cerr << "Warning: Config " << configIdx
                      << " has no valid particles!" << std::endl;
            continue;
        }

        if (config.range.size() < 2) {
            std::cerr << "Error: Config " << configIdx << " range size < 2!"
                      << std::endl;
            continue;
        }

        double min_bin = config.range[0];
        double max_bin = config.range[1];
        int n_bins = config.bins;
        double bin_width = (max_bin - min_bin) / n_bins;

        // 在设备上分配粒子索引数组
        int *device_particle_indices;
        cudaMalloc(&device_particle_indices,
                   particleIndices.size() * sizeof(int));
        cudaMemcpy(device_particle_indices, particleIndices.data(),
                   particleIndices.size() * sizeof(int),
                   cudaMemcpyHostToDevice);

        int *device_group_sizes;
        cudaMalloc(&device_group_sizes, groupSizes.size() * sizeof(int));
        cudaMemcpy(device_group_sizes, groupSizes.data(),
                   groupSizes.size() * sizeof(int), cudaMemcpyHostToDevice);

        // 步骤1：计算所有事件的质量
        double *device_angles;
        cudaMalloc(&device_angles, nEvents * sizeof(double));

        AngleCalculator<<<gridSize, blockSize>>>(
            device_momenta, device_particle_indices, device_group_sizes,
            nParticles, nEvents, device_angles);
        cudaDeviceSynchronize();

        // 步骤2：填充直方图
        double *device_hist_bins;
        cudaMalloc(&device_hist_bins, n_bins * sizeof(double));
        cudaMemset(device_hist_bins, 0, n_bins * sizeof(double));

        fillHistogramKernel<<<gridSize, blockSize>>>(
            device_angles, device_weights, nEvents, device_hist_bins, n_bins,
            min_bin, max_bin, bin_width);
        cudaDeviceSynchronize();

        // 步骤3：将直方图结果复制回主机
        std::vector<double> host_bin_counts(n_bins, 0.0);
        cudaMemcpy(host_bin_counts.data(), device_hist_bins,
                   n_bins * sizeof(double), cudaMemcpyDeviceToHost);

        // 填充TH1F直方图
        for (int bin = 0; bin < n_bins; ++bin) {
            hist->SetBinContent(bin + 1, host_bin_counts[bin]);
        }
        hist->SetBins(n_bins, min_bin, max_bin);

        // 清理设备内存
        cudaFree(device_particle_indices);
        cudaFree(device_angles);
        cudaFree(device_hist_bins);

        // std::cout << "Processed config " << configIdx << " with " << nEvents
        // << " events" << std::endl;
    }
}

__global__ void
fill2DHistogramKernel(const LorentzVector *momenta, const int *particle_indices,
                      const int *group_sizes, const int n_particles,
                      const double *weights, int n_events, double *hist_bins,
                      int x_bins, int y_bins, double x_min, double x_max,
                      double y_min, double y_max, double x_bin_width,
                      double y_bin_width)
{
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (event_idx < n_events) {
        int idx_offset = 0;

        // 计算第一个质量
        LorentzVector p1;
        for (int i = 0; i < group_sizes[0]; ++i) {
            int particle_idx = particle_indices[idx_offset + i];
            p1 = p1 + momenta[event_idx * n_particles + particle_idx];
        }
        idx_offset += group_sizes[0];

        // 计算第二个质量
        LorentzVector p2;
        for (int i = 0; i < group_sizes[1]; ++i) {
            int particle_idx = particle_indices[idx_offset + i];
            p2 = p2 + momenta[event_idx * n_particles + particle_idx];
        }

        double weight = 1.0;
        if (weights != nullptr)
            weight = weights[event_idx];

        if (p1.M2() >= x_min && p1.M2() < x_max && p2.M2() >= y_min &&
            p2.M2() < y_max) {
            int x_bin_idx = static_cast<int>((p1.M2() - x_min) / x_bin_width);
            int y_bin_idx = static_cast<int>((p2.M2() - y_min) / y_bin_width);

            if (x_bin_idx < 0)
                x_bin_idx = 0;
            if (x_bin_idx >= x_bins)
                x_bin_idx = x_bins - 1;
            if (y_bin_idx < 0)
                y_bin_idx = 0;
            if (y_bin_idx >= y_bins)
                y_bin_idx = y_bins - 1;

            // 计算一维索引（行优先）
            int bin_idx = y_bin_idx * x_bins + x_bin_idx;

            // 使用原子操作确保线程安全
            atomicAdd(&hist_bins[bin_idx], weight);
        }
    }
}

void CalculateDalitzHist(LorentzVector *device_momenta,
                         const std::map<std::string, int> &particleToIndex,
                         const std::vector<DalitzHistConfig> &histConfigs,
                         double *device_weights,
                         std::vector<TH2F *> &outputHistograms, int nEvents,
                         int nParticles)
{
    if (outputHistograms.size() != histConfigs.size()) {
        std::cerr << "Error: outputHistograms size (" << outputHistograms.size()
                  << ") does not match histConfigs size (" << histConfigs.size()
                  << ")" << std::endl;
        return;
    }

    if (nEvents == 0) {
        std::cerr << "Warning: No events to process!" << std::endl;
        return;
    }

    // CUDA配置
    int blockSize = 256;
    int gridSize = (nEvents + blockSize - 1) / blockSize;

    for (size_t configIdx = 0; configIdx < histConfigs.size(); ++configIdx) {
        const auto &config = histConfigs[configIdx];
        TH2F *hist = outputHistograms[configIdx];

        if (!hist) {
            std::cerr << "Error: outputHistograms[" << configIdx << "] is null!"
                      << std::endl;
            continue;
        }

        hist->Reset();

        // 获取粒子索引
        std::vector<int> particleIndices;
        std::vector<int> groupSizes;

        for (const auto &particleGroup : config.particles) {
            groupSizes.push_back(particleGroup.size());
            for (const auto &particleName : particleGroup) {
                auto it = particleToIndex.find(particleName);
                if (it == particleToIndex.end()) {
                    std::cerr << "Error: Particle '" << particleName
                              << "' not found in particleToIndex map!"
                              << std::endl;
                    particleIndices.clear();
                    groupSizes.clear();
                    break;
                }
                particleIndices.push_back(it->second);
            }
            if (particleIndices.empty())
                break;
        }

        if (particleIndices.empty()) {
            std::cerr << "Warning: Config " << configIdx
                      << " has no valid particles!" << std::endl;
            continue;
        }

        if (config.range.size() < 2) {
            std::cerr << "Error: Config " << configIdx << " range size < 2!"
                      << std::endl;
            continue;
        }

        double x_min = config.range[0][0];
        double x_max = config.range[0][1];
        double y_min = config.range[1][0];
        double y_max = config.range[1][1];
        int x_bins = config.bins[0];
        int y_bins = config.bins[1];
        double x_bin_width = (x_max - x_min) / x_bins;
        double y_bin_width = (y_max - y_min) / y_bins;

        // 在设备上分配粒子索引数组
        int *device_particle_indices;
        cudaMalloc(&device_particle_indices,
                   particleIndices.size() * sizeof(int));
        cudaMemcpy(device_particle_indices, particleIndices.data(),
                   particleIndices.size() * sizeof(int),
                   cudaMemcpyHostToDevice);

        int *device_group_sizes;
        cudaMalloc(&device_group_sizes, groupSizes.size() * sizeof(int));
        cudaMemcpy(device_group_sizes, groupSizes.data(),
                   groupSizes.size() * sizeof(int), cudaMemcpyHostToDevice);

        // 步骤1：计算所有事件的质量
        // double *device_angles;
        // cudaMalloc(&device_angles, nEvents * sizeof(double));

        // AngleCalculator<<<gridSize, blockSize>>>(device_momenta,
        // device_particle_indices, device_group_sizes, nParticles, nEvents,
        // device_angles); cudaDeviceSynchronize();

        // 步骤2：填充直方图
        double *device_hist_bins;
        cudaMalloc(&device_hist_bins, x_bins * y_bins * sizeof(double));
        cudaMemset(device_hist_bins, 0, x_bins * y_bins * sizeof(double));

        fill2DHistogramKernel<<<gridSize, blockSize>>>(
            device_momenta, device_particle_indices, device_group_sizes,
            nParticles, device_weights, nEvents, device_hist_bins, x_bins,
            y_bins, x_min, x_max, y_min, y_max, x_bin_width, y_bin_width);
        cudaDeviceSynchronize();

        // 步骤3：将直方图结果复制回主机
        std::vector<double> host_bin_counts(x_bins * y_bins, 0.0);
        cudaMemcpy(host_bin_counts.data(), device_hist_bins,
                   x_bins * y_bins * sizeof(double), cudaMemcpyDeviceToHost);

        // 填充TH2F直方图
        hist->SetBins(x_bins, x_min, x_max, y_bins, y_min, y_max);

        for (int y_bin = 0; y_bin < y_bins; ++y_bin) {
            for (int x_bin = 0; x_bin < x_bins; ++x_bin) {
                int bin_idx = y_bin * x_bins + x_bin;
                hist->SetBinContent(x_bin + 1, y_bin + 1,
                                    host_bin_counts[bin_idx]);
            }
        }

        // 清理设备内存
        cudaFree(device_particle_indices);
        cudaFree(device_hist_bins);

        // std::cout << "Processed config " << configIdx << " with " << nEvents
        // << " events" << std::endl;
    }
}
