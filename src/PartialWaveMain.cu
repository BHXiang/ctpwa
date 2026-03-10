// #include <pybind11/pybind11.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <omp.h>
#include <chrono>
#include <random>
#include <fstream>
#include <torch/extension.h>
#include <map>

#include <AmpGen.cuh>
#include <ComputeNLL.cuh>
#include <ComputeGrad.cuh>
#include <ComputeResults.cuh>
#include <Config.cuh>
#include <Figure.cuh>

#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <TObjString.h>
#include <TMatrixD.h>

//////////////////////////////////////////////
struct ChainInfo
{
	std::string name;
	std::map<std::pair<std::string, std::vector<int>>, std::vector<Resonance>> intermediate_resonance_map;
	std::vector<std::vector<Particle>> intermediate_combs;
};

////////////////////////////////////////
std::map<std::string, std::vector<LorentzVector>> readMomentaFromDat(
	const std::vector<std::string> &fileinfo,
	const std::vector<std::string> &particleNames,
	const std::vector<std::string> &particlelists,
	int nEvents = -1)
{
	std::map<std::string, std::vector<LorentzVector>> fullMomenta;
	for (const auto &name : particlelists)
	{
		fullMomenta[name] = std::vector<LorentzVector>();
	}

	// 检查输入参数
	if (fileinfo.size() < 2)
	{
		std::cerr << "Error: fileinfo must contain at least file type and filename" << std::endl;
		return fullMomenta;
	}

	std::string fileType = fileinfo[0];
	std::string filename = fileinfo[1];

	std::unordered_set<std::string> particleNameSet(particleNames.begin(), particleNames.end());
	std::string initialName;
	bool foundParticle = false;

	for (const auto &name : particlelists)
	{
		if (particleNameSet.find(name) == particleNameSet.end())
		{
			if (!foundParticle)
			{
				initialName = name;
				foundParticle = true;
			}
			else
			{
				std::cerr << "Error: Found multiple particles in particlelists not present in particleNames" << std::endl;
				return fullMomenta;
			}
		}
	}

	// 处理DAT文件
	if (fileType == "dat")
	{
		std::ifstream file(filename);
		if (!file.is_open())
		{
			std::cerr << "Error: Cannot open file " << filename << std::endl;
			return fullMomenta;
		}

		std::string line;
		int eventCount = 0;
		int lineCount = 0;
		int particlesPerEvent = particleNames.size();

		while (std::getline(file, line))
		{
			if (line.empty())
				continue;

			std::istringstream iss(line);
			double E, px, py, pz;

			if (iss >> E >> px >> py >> pz)
			{
				// 根据行号确定粒子类型
				int particleIndex = lineCount % particlesPerEvent;
				const std::string &particleName = particleNames[particleIndex];

				fullMomenta[particleName].emplace_back(E, px, py, pz);
				lineCount++;

				// 每读完一组粒子表示完成一个事件
				if (particleIndex == particlesPerEvent - 1)
				{
					LorentzVector initialMomentum(0, 0, 0, 0);
					for (const auto &name : particleNames)
					{
						initialMomentum = initialMomentum + fullMomenta[name].back();
					}

					fullMomenta[initialName].emplace_back(initialMomentum);

					eventCount++;

					// 如果指定了事件数并且已达到，则停止读取
					if (nEvents > 0 && eventCount >= nEvents)
					{
						break;
					}
				}
			}
			else
			{
				std::cerr << "Warning: Invalid line format: " << line << std::endl;
			}
		}

		file.close();
	}
	// 处理ROOT文件
	else if (fileType == "ROOT" || fileType == "root")
	{
		// #ifdef USE_ROOT
		if (fileinfo.size() < 3)
		{
			std::cerr << "Error: For ROOT files, fileinfo must contain at least file type, filename and TTree name" << std::endl;
			return fullMomenta;
		}

		std::string treeName = fileinfo[2];

		// 打开ROOT文件
		TFile *file = TFile::Open(filename.c_str(), "READ");
		if (!file || file->IsZombie())
		{
			std::cerr << "Error: Cannot open ROOT file " << filename << std::endl;
			return fullMomenta;
		}

		// 获取TTree
		TTree *tree = (TTree *)file->Get(treeName.c_str());
		if (!tree)
		{
			std::cerr << "Error: Cannot find TTree " << treeName << " in file " << filename << std::endl;
			file->Close();
			delete file;
			return fullMomenta;
		}

		// 准备读取TLorentzVector的分支
		std::vector<TLorentzVector *> particleLV(particleNames.size());
		std::vector<std::string> branchNames;

		// 如果提供了分支名，使用提供的分支名
		if (fileinfo.size() >= 4 + particleNames.size())
		{
			for (size_t i = 0; i < particleNames.size(); ++i)
			{
				branchNames.push_back(fileinfo[3 + i]);
			}
		}
		// 否则使用粒子名作为分支名
		else
		{
			branchNames = particleNames;
		}

		// 设置分支地址
		for (size_t i = 0; i < particleNames.size(); ++i)
		{
			particleLV[i] = new TLorentzVector();
			tree->SetBranchAddress(branchNames[i].c_str(), &particleLV[i]);
		}

		// 读取事件
		Long64_t nEntries = tree->GetEntries();
		if (nEvents > 0 && nEvents < nEntries)
		{
			nEntries = nEvents;
		}

		for (Long64_t iEvent = 0; iEvent < nEntries; ++iEvent)
		{
			tree->GetEntry(iEvent);

			// 读取每个粒子的四动量
			for (size_t i = 0; i < particleNames.size(); ++i)
			{
				const std::string &particleName = particleNames[i];
				TLorentzVector *lv = particleLV[i];

				// 转换为你的LorentzVector类型
				// 假设你的LorentzVector构造函数接受(E, px, py, pz)
				fullMomenta[particleName].emplace_back(
					lv->E(), lv->Px(), lv->Py(), lv->Pz());
			}

			// 计算初始粒子的四动量
			LorentzVector initialMomentum(0, 0, 0, 0);
			for (const auto &name : particleNames)
			{
				initialMomentum = initialMomentum + fullMomenta[name].back();
			}
			fullMomenta[initialName].emplace_back(initialMomentum);
		}

		// 清理内存
		for (auto lv : particleLV)
		{
			delete lv;
		}

		file->Close();
		delete file;
		// #else
		// std::cerr << "Error: ROOT support not compiled in. Please define USE_ROOT and link with ROOT libraries." << std::endl;
		// return fullMomenta;
		// #endif
	}
	else
	{
		std::cerr << "Error: Unknown file type: " << fileType << std::endl;
		return fullMomenta;
	}

	return fullMomenta;
}

double *readWeightsFromFile(const std::vector<std::string> &fileinfo, int length)
{
	// 检查输入参数
	if (fileinfo.size() < 2)
	{
		std::cerr << "Error: fileinfo must contain at least file type and filename" << std::endl;
		return nullptr;
	}

	std::string fileType = fileinfo[0];
	std::string filename = fileinfo[1];

	std::vector<double> weights;

	// 处理DAT文件
	if (fileType == "dat")
	{
		std::ifstream file(filename);
		if (!file.is_open())
		{
			std::cerr << "Error: Cannot open file " << filename << std::endl;
			return nullptr;
		}

		double weight;
		while (file >> weight)
		{
			weights.push_back(weight);
		}

		file.close();
	}
	// 处理ROOT文件
	else if (fileType == "ROOT" || fileType == "root")
	{
		// #ifdef USE_ROOT
		if (fileinfo.size() < 4)
		{
			std::cerr << "Error: For ROOT files, fileinfo must contain at least file type, filename, TTree name and weight branch name" << std::endl;
			return nullptr;
		}

		std::string treeName = fileinfo[2];
		std::string branchName = fileinfo[3];

		// 打开ROOT文件
		TFile *file = TFile::Open(filename.c_str(), "READ");
		if (!file || file->IsZombie())
		{
			std::cerr << "Error: Cannot open ROOT file " << filename << std::endl;
			return nullptr;
		}

		// 获取TTree
		TTree *tree = (TTree *)file->Get(treeName.c_str());
		if (!tree)
		{
			std::cerr << "Error: Cannot find TTree " << treeName << " in file " << filename << std::endl;
			file->Close();
			delete file;
			return nullptr;
		}

		// 设置权重分支
		double weight = 0.0;
		tree->SetBranchAddress(branchName.c_str(), &weight);

		// 读取所有事件的权重
		Long64_t nEntries = tree->GetEntries();
		for (Long64_t iEvent = 0; iEvent < nEntries; ++iEvent)
		{
			tree->GetEntry(iEvent);
			weights.push_back(weight);
		}

		file->Close();
		delete file;
		// #else
		// std::cerr << "Error: ROOT support not compiled in. Please define USE_ROOT and link with ROOT libraries." << std::endl;
		// return nullptr;
		// #endif
	}
	else
	{
		std::cerr << "Error: Unknown file type: " << fileType << std::endl;
		return nullptr;
	}

	// 检查权重数量
	if (length > 0 && weights.size() != static_cast<size_t>(length))
	{
		std::cerr << "Error: Weights size " << weights.size() << " does not match expected length " << length << std::endl;
		// 可以根据需求决定是否返回nullptr
		// return nullptr;
	}

	// 如果length为-1或0，使用实际读取的权重数量
	if (length <= 0)
	{
		length = weights.size();
	}

	// 分配设备内存并复制数据
	double *d_weights = nullptr;
	cudaError_t cudaStatus = cudaMalloc(&d_weights, length * sizeof(double));
	// 确保weights向量有足够的元素
	if (weights.size() < static_cast<size_t>(length))
	{
		std::cerr << "Warning: Not enough weights in file. Padding with zeros." << std::endl;
		weights.resize(length, 0.0);
	}

	cudaStatus = cudaMemcpy(d_weights, weights.data(), length * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "Error: cudaMemcpy failed for weights: " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaFree(d_weights);
		return nullptr;
	}

	return d_weights;
}

LorentzVector *convertToLorentzVector(const std::map<std::string, std::vector<LorentzVector>> &finalMomenta, const std::map<std::string, int> &particleToIndex)
{
	// 获取事件数量和粒子数量
	int n_events = finalMomenta.begin()->second.size();
	int n_particles = particleToIndex.size();

	// 在主机端分配所有粒子的四动量数组
	std::vector<LorentzVector> host_momenta(n_events * n_particles);
	std::fill(host_momenta.begin(), host_momenta.end(), LorentzVector());

	// 创建粒子计算状态标记
	std::vector<bool> particle_calculated(n_particles, false);

	// 第一步：将末态粒子的四动量复制到对应位置
	for (const auto &particle_momenta : finalMomenta)
	{
		const std::string &particle_name = particle_momenta.first;
		const std::vector<LorentzVector> &momenta_vec = particle_momenta.second;

		auto it = particleToIndex.find(particle_name);
		if (it != particleToIndex.end())
		{
			int particle_idx = it->second;
			for (int event_idx = 0; event_idx < n_events; ++event_idx)
			{
				host_momenta[event_idx * n_particles + particle_idx] = momenta_vec[event_idx];
			}
			particle_calculated[particle_idx] = true;
		}
	}

	// 第二步：将数据复制到设备
	LorentzVector *d_momenta;
	cudaMalloc(&d_momenta, host_momenta.size() * sizeof(LorentzVector));
	cudaMemcpy(d_momenta, host_momenta.data(), host_momenta.size() * sizeof(LorentzVector), cudaMemcpyHostToDevice);

	return d_momenta;
}

//////////////////////////////////////////////////////////////
/// NLLFunction 类定义
///////////////////////////////////////////////////////////////
class NLLFunction : public torch::autograd::Function<NLLFunction>
{
private:
	// 私有成员变量，存储约束信息
	static std::vector<std::vector<int>> con_trans_id_;
	static std::vector<std::vector<std::complex<double>>> con_trans_values_;
	static bool constraints_initialized_;

public:
	static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
								 torch::Tensor &vector,
								 int n_gls_,
								 int n_polar_,
								 const cuComplex *data_fix_,
								 int data_length,
								 const cuComplex *phsp_fix_,
								 int phsp_length,
								 const cuComplex *bkg_fix_,
								 const double *bkg_weights_,
								 int bkg_length)
	{
		TORCH_CHECK(vector.is_cuda(), "[NLLForward] vector must be on CUDA");
		TORCH_CHECK(vector.dtype() == c10::kComplexFloat, "[NLLForward] vector must be complex64");

		// 检查约束是否已初始化
		TORCH_CHECK(constraints_initialized_, "[NLLForward] Constraints not initialized. Call setConstraints() first.");

		// 获取当前设备并设置
		const int target_dev = vector.get_device();
		torch::Device dev(torch::kCUDA, target_dev);

		// 延长vector以处理约束
		torch::Tensor extended_vector = extendVectorWithConstraints(vector, dev);
		const int extended_n_gls = extended_vector.numel();

		// 后续逻辑（MC因子计算等）
		cuComplex *d_B = nullptr;
		double *d_mc_amp = nullptr;
		cudaMalloc(&d_B, phsp_length * sizeof(cuComplex));
		cudaMalloc(&d_mc_amp, sizeof(double));

		// 注意：这里使用了 c10::complex 类型
		computePHSPfactor(phsp_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), d_B, d_mc_amp, phsp_length, extended_n_gls);

		double h_phsp_factor;
		cudaMemcpy(&h_phsp_factor, d_mc_amp, sizeof(double), cudaMemcpyDeviceToHost);
		h_phsp_factor = h_phsp_factor / static_cast<double>(phsp_length / n_polar_);

		// std::cout << "[NLLForward] PHSP factor: " << h_phsp_factor << " Nphsp: " << phsp_length / n_polar_ << std::endl;

		// NLL计算
		cuComplex *d_S = nullptr;
		cuComplex *d_Q = nullptr;
		double *d_data_nll = nullptr;
		const int Q_numel = data_length / n_polar_;
		cudaMalloc(&d_S, data_length * sizeof(cuComplex));
		cudaMalloc(&d_Q, Q_numel * sizeof(cuComplex));
		cudaMalloc(&d_data_nll, sizeof(double));

		computeNll(data_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), nullptr, d_S, d_Q, d_data_nll, data_length, extended_n_gls, n_polar_, h_phsp_factor);

		double h_data_nll;
		cudaMemcpy(&h_data_nll, d_data_nll, sizeof(double), cudaMemcpyDeviceToHost);

		// bkg部分
		cuComplex *d_bkg_S = nullptr;
		cuComplex *d_bkg_Q = nullptr;
		double *d_bkg_nll = nullptr;
		const int bkg_Q_numel = bkg_length / n_polar_;
		cudaMalloc(&d_bkg_S, bkg_length * sizeof(cuComplex));
		cudaMalloc(&d_bkg_Q, bkg_Q_numel * sizeof(cuComplex));
		cudaMalloc(&d_bkg_nll, sizeof(double));
		double h_bkg_nll = 0.0;
		if (bkg_fix_ != nullptr && bkg_length > 0)
		{
			computeNll(bkg_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), bkg_weights_, d_bkg_S, d_bkg_Q, d_bkg_nll, bkg_length, extended_n_gls, n_polar_, h_phsp_factor);

			cudaMemcpy(&h_bkg_nll, d_bkg_nll, sizeof(double), cudaMemcpyDeviceToHost);
		}

		// 保存反向传播变量
		ctx->saved_data["target_dev"] = target_dev;
		ctx->saved_data["n_polar"] = n_polar_;
		ctx->saved_data["h_phsp_factor"] = h_phsp_factor * static_cast<double>(phsp_length / n_polar_);
		ctx->saved_data["n_gls"] = n_gls_;
		ctx->saved_data["extended_n_gls"] = extended_n_gls;
		ctx->saved_data["data_length"] = data_length;
		ctx->saved_data["phsp_length"] = phsp_length;
		ctx->saved_data["bkg_length"] = bkg_length;

		// 保存显存指针
		ctx->saved_data["data_fix_ptr"] = reinterpret_cast<int64_t>(data_fix_);
		ctx->saved_data["phsp_fix_ptr"] = reinterpret_cast<int64_t>(phsp_fix_);
		ctx->saved_data["bkg_fix_ptr"] = reinterpret_cast<int64_t>(bkg_fix_);
		ctx->saved_data["bkg_weights_ptr"] = reinterpret_cast<int64_t>(bkg_weights_);
		ctx->saved_data["d_B_ptr"] = reinterpret_cast<int64_t>(d_B);
		ctx->saved_data["d_S_ptr"] = reinterpret_cast<int64_t>(d_S);
		ctx->saved_data["d_Q_ptr"] = reinterpret_cast<int64_t>(d_Q);
		ctx->saved_data["d_bkg_S_ptr"] = reinterpret_cast<int64_t>(d_bkg_S);
		ctx->saved_data["d_bkg_Q_ptr"] = reinterpret_cast<int64_t>(d_bkg_Q);
		ctx->saved_data["d_bkg_nll_ptr"] = reinterpret_cast<int64_t>(d_bkg_nll);

		ctx->save_for_backward({vector, extended_vector});

		// 释放临时内存
		cudaFree(d_mc_amp);

		return torch::tensor({h_data_nll - h_bkg_nll}, torch::kDouble).to(dev);
	}

	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
												 const torch::autograd::tensor_list &grad_outputs)
	{
		const int target_dev = ctx->saved_data["target_dev"].toInt();

		// 从 saved_data 获取参数
		const int n_polar = ctx->saved_data["n_polar"].toInt();
		const double h_phsp_factor = ctx->saved_data["h_phsp_factor"].toDouble();
		const int n_gls = ctx->saved_data["n_gls"].toInt();
		const int extended_n_gls = ctx->saved_data["extended_n_gls"].toInt();
		const int data_length = ctx->saved_data["data_length"].toInt();
		const int phsp_length = ctx->saved_data["phsp_length"].toInt();
		const int bkg_length = ctx->saved_data["bkg_length"].toInt();

		// 从 saved_data 获取显存指针
		cuComplex *d_B = reinterpret_cast<cuComplex *>(ctx->saved_data["d_B_ptr"].toInt());
		cuComplex *data_fix = reinterpret_cast<cuComplex *>(ctx->saved_data["data_fix_ptr"].toInt());
		cuComplex *phsp_fix = reinterpret_cast<cuComplex *>(ctx->saved_data["phsp_fix_ptr"].toInt());
		cuComplex *d_S = reinterpret_cast<cuComplex *>(ctx->saved_data["d_S_ptr"].toInt());
		cuComplex *d_Q = reinterpret_cast<cuComplex *>(ctx->saved_data["d_Q_ptr"].toInt());

		cuComplex *bkg_fix = reinterpret_cast<cuComplex *>(ctx->saved_data["bkg_fix_ptr"].toInt());
		double *bkg_weights = reinterpret_cast<double *>(ctx->saved_data["bkg_weights_ptr"].toInt());
		cuComplex *d_bkg_S = reinterpret_cast<cuComplex *>(ctx->saved_data["d_bkg_S_ptr"].toInt());
		cuComplex *d_bkg_Q = reinterpret_cast<cuComplex *>(ctx->saved_data["d_bkg_Q_ptr"].toInt());

		// 获取保存的变量
		const auto saved = ctx->get_saved_variables();
		const auto &original_vector = saved[0];
		const auto &extended_vector = saved[1];

		// 计算扩展向量的梯度
		cuComplex *d_extended_grad = nullptr;
		cudaMalloc(&d_extended_grad, extended_n_gls * sizeof(cuComplex));

		cublasHandle_t cublas_handle;
		cublasCreate(&cublas_handle);
		compute_gradient(data_fix, phsp_fix, d_S, d_Q, d_B, nullptr, h_phsp_factor, extended_n_gls, data_length / n_polar, n_polar, phsp_length, d_extended_grad, cublas_handle);

		// 如果有背景数据，减去背景NLL的梯度
		if (bkg_fix != nullptr && bkg_length > 0)
		{
			cuComplex *d_bkg_extended_grad = nullptr;
			cudaMalloc(&d_bkg_extended_grad, extended_n_gls * sizeof(cuComplex));

			cudaMemset(d_bkg_extended_grad, 0, extended_n_gls * sizeof(cuComplex));

			compute_gradient(bkg_fix, phsp_fix, d_bkg_S, d_bkg_Q, d_B, bkg_weights, h_phsp_factor, extended_n_gls, bkg_length / n_polar, n_polar, phsp_length, d_bkg_extended_grad, cublas_handle);

			const cuComplex minus_one = make_cuComplex(-1.0f, 0.0f);
			cublasCaxpy(cublas_handle, extended_n_gls,
						&minus_one, d_bkg_extended_grad, 1,
						d_extended_grad, 1);

			cudaFree(d_bkg_extended_grad);
			cudaFree(d_bkg_S);
			cudaFree(d_bkg_Q);
		}

		// 将扩展梯度复制到torch张量
		torch::Tensor extended_grad = torch::empty({extended_n_gls}, torch::kComplexFloat).to(original_vector.device());
		cudaMemcpy(extended_grad.data_ptr(), d_extended_grad, extended_n_gls * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

		// 合并梯度（考虑约束关系）
		torch::Tensor grad_vector = mergeGradientsWithConstraints(extended_grad, original_vector.numel());

		// 清理内存
		cudaFree(d_extended_grad);
		cudaFree(d_B);
		cudaFree(d_S);
		cudaFree(d_Q);
		cublasDestroy(cublas_handle);

		return {grad_vector, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
	}

	// 设置约束的静态方法
	static void setConstraints(const std::vector<std::vector<int>> &con_trans_id,
							   const std::vector<std::vector<std::complex<double>>> &con_trans_values)
	{
		con_trans_id_ = con_trans_id;
		con_trans_values_ = con_trans_values;
		constraints_initialized_ = true;
	}

	// 清除约束的静态方法
	// static void clearConstraints()
	// {
	// 	con_trans_id_.clear();
	// 	con_trans_values_.clear();
	// 	constraints_initialized_ = false;
	// }

	static torch::Tensor extendVectorWithConstraints(const torch::Tensor &vector,
													 const torch::Device &device)
	{
		TORCH_CHECK(vector.is_complex(), "Input vector must be complex type");
		TORCH_CHECK(vector.dim() == 1, "Input vector must be 1-dimensional");

		const int original_size = vector.numel();
		int extended_size = original_size;

		// 找到最大ID以确定扩展大小
		for (const auto &vecid : con_trans_id_)
		{
			if (!vecid.empty())
			{
				auto max_it = std::max_element(vecid.begin(), vecid.end());
				extended_size = std::max(extended_size, *max_it + 1);
			}
		}

		if (extended_size == original_size)
		{
			return vector.clone();
		}

		// 创建扩展后的向量
		torch::TensorOptions options = torch::TensorOptions()
										   .dtype(torch::kComplexFloat)
										   .device(device);

		torch::Tensor extended_vector = torch::zeros({extended_size}, options);

		// 方法1：使用 PyTorch 的索引操作（在 GPU 上）
		// 创建索引，选择原始部分
		torch::Tensor indices = torch::arange(0, original_size, torch::kLong).to(device);
		extended_vector.index_copy_(0, indices, vector);

		// 在 GPU 上处理约束
		for (size_t i = 0; i < con_trans_id_.size(); ++i)
		{
			const auto &vecid = con_trans_id_[i];
			const auto &values = con_trans_values_[i];

			if (vecid.empty() || values.empty() || vecid.size() != values.size())
			{
				continue;
			}

			// 找到原始ID（最小值）
			auto min_it = std::min_element(vecid.begin(), vecid.end());
			int origin_idx = std::distance(vecid.begin(), min_it);
			int origin_id = vecid[origin_idx];

			// 确保原始ID有效
			if (origin_id < 0 || origin_id >= original_size)
			{
				continue;
			}

			// 获取原始ID对应的系数
			std::complex<double> origin_coeff = values[origin_idx];
			double origin_coeff_real = std::real(origin_coeff);
			double origin_coeff_imag = std::imag(origin_coeff);

			// 检查分母不为零
			if (std::abs(origin_coeff_real) < 1e-10 || std::abs(origin_coeff_imag) < 1e-10)
			{
				std::cerr << "Warning: origin coefficient too small, skipping constraint group " << i << std::endl;
				continue;
			}

			// 为每个扩展ID设置值
			for (size_t j = 0; j < vecid.size(); ++j)
			{
				if (j == origin_idx)
					continue; // 跳过原始ID

				int extended_id = vecid[j];

				// 确保扩展ID有效且不超过values数组的大小
				if (extended_id >= 0 && extended_id < extended_size && j < values.size())
				{
					std::complex<double> ext_coeff = values[j];
					double ext_coeff_real = std::real(ext_coeff);
					double ext_coeff_imag = std::imag(ext_coeff);

					// 计算系数比例（直接在 GPU 上）
					float real_ratio = static_cast<float>(ext_coeff_real / origin_coeff_real);
					float imag_ratio = static_cast<float>(ext_coeff_imag / origin_coeff_imag);

					// 在 GPU 上执行线性变换
					// extended_vector[extended_id] = real_ratio * real_part + imag_ratio * imag_part

					// 获取原始向量的值
					torch::Tensor origin_value = vector[origin_id];

					// 计算实部和虚部
					torch::Tensor real_part = (origin_value + torch::conj(origin_value)) / 2.0f;
					torch::Tensor imag_part = (origin_value - torch::conj(origin_value)) / (2.0f * c10::complex<float>(0, 1));

					// 计算扩展值
					torch::Tensor extended_real = real_ratio * real_part;
					torch::Tensor extended_imag = imag_ratio * imag_part;

					// 合并实部和虚部
					torch::Tensor extended_value = extended_real + c10::complex<float>(0, 1) * extended_imag;

					// 赋值
					extended_vector[extended_id] = extended_value;
				}
			}
		}

		return extended_vector;
	}

private:
	static torch::Tensor mergeGradientsWithConstraints(
		const torch::Tensor &extended_grad,
		int original_size)
	{
		torch::Device device = extended_grad.device();
		torch::Tensor grad_vector = torch::zeros({original_size}, torch::kComplexFloat).to(device);

		// 复制原始元素的梯度
		if (original_size > 0)
		{
			grad_vector.copy_(extended_grad.slice(0, 0, original_size));
		}

		// 如果没有约束，直接返回
		if (con_trans_id_.empty())
		{
			return grad_vector;
		}

		// 对于每个约束组
		for (size_t group_idx = 0; group_idx < con_trans_id_.size(); ++group_idx)
		{
			const auto &vecid = con_trans_id_[group_idx];
			const auto &values = con_trans_values_[group_idx];

			if (vecid.empty() || values.empty() || vecid.size() != values.size())
			{
				continue;
			}

			// 找到原始ID（最小值）
			auto min_it = std::min_element(vecid.begin(), vecid.end());
			int origin_idx = std::distance(vecid.begin(), min_it);
			int origin_id = vecid[origin_idx];

			if (origin_id < 0 || origin_id >= original_size)
			{
				continue;
			}

			std::complex<double> origin_coeff = values[origin_idx];
			double origin_coeff_real = std::real(origin_coeff);
			double origin_coeff_imag = std::imag(origin_coeff);

			if (std::abs(origin_coeff_real) < 1e-10 || std::abs(origin_coeff_imag) < 1e-10)
			{
				continue;
			}

			// 收集该原始元素对应的所有扩展元素
			std::vector<int> ext_indices;
			std::vector<float> real_ratios, imag_ratios;

			for (size_t j = 0; j < vecid.size(); ++j)
			{
				if (j == origin_idx)
					continue;

				int extended_id = vecid[j];
				if (extended_id < 0 || extended_id >= extended_grad.numel() || j >= values.size())
				{
					continue;
				}

				std::complex<double> ext_coeff = values[j];
				real_ratios.push_back(static_cast<float>(std::real(ext_coeff) / origin_coeff_real));
				imag_ratios.push_back(static_cast<float>(std::imag(ext_coeff) / origin_coeff_imag));
				ext_indices.push_back(extended_id);
			}

			if (ext_indices.empty())
				continue;

			// 转换为张量
			torch::Tensor ext_idx_tensor = torch::tensor(ext_indices, torch::kLong).to(device);
			torch::Tensor real_ratio_tensor = torch::tensor(real_ratios, torch::kFloat).to(device);
			torch::Tensor imag_ratio_tensor = torch::tensor(imag_ratios, torch::kFloat).to(device);

			// 获取扩展元素的梯度
			torch::Tensor ext_grads = extended_grad.index_select(0, ext_idx_tensor);

			// 分离实部和虚部
			torch::Tensor ext_grad_real = (ext_grads + torch::conj(ext_grads)) / 2.0f;
			torch::Tensor ext_grad_imag = (ext_grads - torch::conj(ext_grads)) / (2.0f * c10::complex<float>(0, 1));

			// 计算总贡献
			torch::Tensor total_contrib = (real_ratio_tensor * ext_grad_real +
										   c10::complex<float>(0, 1) * imag_ratio_tensor * ext_grad_imag)
											  .sum();

			// 累加到原始元素
			grad_vector[origin_id] = grad_vector[origin_id] + total_contrib;
		}

		return grad_vector;
	}
};

// 初始化静态成员变量
std::vector<std::vector<int>> NLLFunction::con_trans_id_;
std::vector<std::vector<std::complex<double>>> NLLFunction::con_trans_values_;
bool NLLFunction::constraints_initialized_ = false;

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
class analysis
{
public:
	analysis(const std::string &config_file = "config.yml") : config_parser_(config_file), n_gls_(0), n_polar_(0), data_fix_(nullptr), data_length(0), phsp_fix_(nullptr), phsp_length(0), bkg_fix_(nullptr), bkg_length(0)
	{
		initialize();
	}

	// 析构函数，用于释放 CUDA 内存
	~analysis()
	{
		if (data_fix_ != nullptr)
		{
			cudaFree(data_fix_);
			data_fix_ = nullptr;
		}
		if (phsp_fix_ != nullptr)
		{
			cudaFree(phsp_fix_);
			phsp_fix_ = nullptr;
		}
		if (bkg_fix_ != nullptr)
		{
			cudaFree(bkg_fix_);
			bkg_fix_ = nullptr;
		}
	}

	torch::Tensor getNLL(torch::Tensor &vector)
	{
		return NLLFunction::apply(vector, n_gls_, n_polar_, data_fix_, data_length, phsp_fix_, phsp_length, bkg_fix_, bkg_weights_, bkg_length);
		// return NLLFunction::apply(vector, n_gls_, n_polar_, data_fix_, data_length, phsp_fix_, phsp_length, bkg_fix_, bkg_length, con_trans_id_, con_trans_values_);
		// return NLLFunction::apply(vector, n_gls_, n_polar_, data_fix_, data_length, phsp_fix_, phsp_length, bkg_fix_, bkg_length, conjugate_pairs_);
		// return NLLFunction::apply(vector, n_gls_, n_polar_, data_fix_, data_length, phsp_fix_, phsp_length, bkg_fix_, bkg_length);
	}

	int getNVector() const
	{
		return n_gls_ - con_trans_id_.size();
	}

	torch::Tensor getSLVectors() const
	{
		torch::Device dev(torch::kCUDA, 0);
		torch::TensorOptions options = torch::TensorOptions()
										   .dtype(torch::kInt)
										   .device(dev);
		return torch::tensor(nSLvectors_, options);
	}

	void writeResult(torch::Tensor &vector, const std::string &filename, const int is_saved_weight = 0)
	{
		TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
		TORCH_CHECK(vector.dtype() == torch::kComplexFloat, "vector must be complex128");

		const int original_size = vector.numel();
		int extended_size = original_size;

		// const int target_dev = vector.get_device();
		torch::Device dev(torch::kCUDA, vector.get_device());

		torch::Tensor extended_vector = NLLFunction::extendVectorWithConstraints(vector, dev);

		const int target_dev = vector.get_device();
		cudaSetDevice(target_dev);

		const int n_events = phsp_length / n_polar_; // 事件数量
		double *d_final_result;
		double *d_partial_result;
		// double *d_partial_sum;

		// 分配设备内存
		cudaMalloc(&d_final_result, n_events * sizeof(double));
		int npartials = nSLvectors_.size();
		cudaMalloc(&d_partial_result, n_events * npartials * sizeof(double));
		// cudaMalloc(&d_partial_sum, npartials * sizeof(double));

		// 分配nSLvectors_的设备内存
		int *d_nSLvectors;
		cudaMalloc(&d_nSLvectors, nSLvectors_.size() * sizeof(int));
		cudaMemcpy(d_nSLvectors, nSLvectors_.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);
		double *d_total_integral;
		cudaMalloc(&d_total_integral, sizeof(double));
		cudaMemset(d_total_integral, 0, sizeof(double));

		// 分配干涉矩阵和事件干涉项的设备内存
		double *d_interference_matrix;
		cudaMalloc(&d_interference_matrix, npartials * npartials * sizeof(double));
		cudaMemset(d_interference_matrix, 0, npartials * npartials * sizeof(double));
		// double *d_event_interference;
		// cudaMalloc(&d_event_interference, n_events * npartials * npartials * sizeof(double));
		// cudaMemset(d_event_interference, 0, n_events * npartials * npartials * sizeof(double));

		// 计算权重
		// computeResults(phsp_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), d_final_result, d_total_integral, d_partial_result, d_interference_matrix, nullptr, d_nSLvectors, npartials, n_events, n_gls_, n_polar_);
		computeResults(phsp_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), d_final_result, d_total_integral, d_partial_result, d_interference_matrix, d_nSLvectors, npartials, n_events, n_gls_, n_polar_);
		// computeWeightResult(phsp_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), d_final_result, d_total_integral, d_partial_result, d_nSLvectors, npartials, n_events, n_gls_, n_polar_);

		double *h_total_results = new double[n_events];
		cudaMemcpy(h_total_results, d_final_result, n_events * sizeof(double), cudaMemcpyDeviceToHost);
		double *h_partial_results = new double[n_events * npartials];
		cudaMemcpy(h_partial_results, d_partial_result, n_events * npartials * sizeof(double), cudaMemcpyDeviceToHost);
		// double *h_partial_sums = new double[npartials];
		// cudaMemcpy(h_partial_sums, d_partial_sum, npartials * sizeof(double), cudaMemcpyDeviceToHost);
		double h_phsp_integral;
		cudaMemcpy(&h_phsp_integral, d_total_integral, sizeof(double), cudaMemcpyDeviceToHost);
		double *h_interference_matrix = new double[npartials * npartials];
		cudaMemcpy(h_interference_matrix, d_interference_matrix, npartials * npartials * sizeof(double), cudaMemcpyDeviceToHost);

		// bkg weights积分
		double h_bkg_integral = 0.0;
		if (bkg_weights_ != nullptr && bkg_length > 0)
		{
			thrust::device_ptr<double> d_ptr(bkg_weights_);
			std::cout << "Calculating background integral with " << bkg_length / n_polar_ << " events..." << std::endl;
			h_bkg_integral = thrust::reduce(d_ptr, d_ptr + bkg_length / n_polar_);
			// h_bkg_integral = thrust::reduce(d_ptr, d_ptr + bkg_length / n_polar_, 0.0, thrust::plus<double>());
		}

		///////////////////////////////////////////////////
		// truth phase space for branching fraciton
		///////////////////////////////////////////////////
		// {
		// 	const auto &data_files = config_parser_.getDataFiles();
		// 	const auto &data_order = config_parser_.getDataOrder();

		// 	std::vector<std::string> particles_names;
		// 	for (const auto &particle : particles_)
		// 	{
		// 		particles_names.push_back(particle.name);
		// 	}

		// 	Vp4_truth_ = readWeightsFromFile(data_files.at("phsp_truth"), data_order, particles_names);
		// 	std::cout << "Phase space truth events: " << Vp4_truth_.begin()->second.size() << std::endl;
		// 	std::cout << "Calculating phase space truth amplitudes..." << std::endl;
		// 	cuComplex *truth_amplitude = calculateAmplitudes(Vp4_truth_);
		// 	truth_length = Vp4_truth_.begin()->second.size() * n_polar_;
		// }

		// // 写入文件
		// std::ofstream outfile("total_weights.dat");
		// if (outfile.is_open())
		// {
		// 	// for (const auto &weight : h_row_results)
		// 	for (int i = 0; i < n_events; ++i)
		// 	{
		// 		outfile << h_total_results[i] << std::endl;
		// 	}
		// 	outfile.close();
		// 	std::cout << "Weights written to total_weights.dat" << std::endl;
		// }
		// else
		// {
		// 	std::cerr << "Unable to open file: total_weights.dat" << std::endl;
		// }

		int dataIntegral = data_length / n_polar_;
		if (bkg_fix_ != nullptr && bkg_length > 0)
		{
			if (h_bkg_integral > 0)
			{
				dataIntegral -= h_bkg_integral;
			}
			else
			{
				dataIntegral -= bkg_length / n_polar_;
			}
		}
		double normFactor = static_cast<double>(dataIntegral) / h_phsp_integral;

		// 创建 ROOT 文件
		TFile *rootFile = new TFile(filename.c_str(), "RECREATE");

		TTree *legend = new TTree("legends", "Amplitude Legends");
		legend->Branch("legend", &legends_);
		legend->Fill();
		legend->Write();
		delete legend;

		// 写入干涉矩阵
		if (h_interference_matrix != nullptr)
		{
			TMatrixD interferenceMatrix(npartials, npartials);
			for (int i = 0; i < npartials; ++i)
			{
				for (int j = i; j < npartials; ++j)
				{
					int idx = i * npartials - i * (i - 1) / 2 + (j - i);
					interferenceMatrix[i][j] = h_interference_matrix[idx];
				}
			}

			interferenceMatrix.Write("interference");
		}

		if (is_saved_weight == 1)
		{
			TTree *phspTree = new TTree("saved_weight", "fitting result weights");

			// 添加权重分支
			double total_weight;
			phspTree->Branch("totalweight", &total_weight);

			// 为每个部分波创建分支
			std::vector<double> partial_weights(npartials);
			for (int i = 0; i < npartials; ++i)
			{
				std::string branch_name = "weight_" + resonance_names_[i];
				phspTree->Branch(branch_name.c_str(), &partial_weights[i]);
			}

			// 填充 phsp tree
			for (int i = 0; i < n_events; ++i)
			{
				// 设置权重
				total_weight = h_total_results[i] / h_phsp_integral * static_cast<double>(dataIntegral);
				// std::cout << "Event " << i << ": Total Weight = " << total_weight << std::endl;
				for (int j = 0; j < npartials; ++j)
				{
					partial_weights[j] = h_partial_results[i * npartials + j] * normFactor;
					// std::cout << "  Partial Weight " << j << " = " << partial_weights[j] << std::endl;
				}

				phspTree->Fill();
			}

			// 干涉项分支
			// double *h_event_interference = new double[n_events * npartials * npartials];
			// cudaMemcpy(h_event_interference, d_event_interference, n_events * npartials * npartials * sizeof(double), cudaMemcpyDeviceToHost);
			// std::vector<std::vector<double>> interference_terms(npartials, std::vector<double>(npartials));
			// for (int i = 0; i < npartials; ++i)
			// {
			// 	for (int j = 0; j < npartials; ++j)
			// 	{
			// 		std::string branch_name = "inter_" + resonance_names_[i] + "_" + resonance_names_[j];
			// 		phspTree->Branch(branch_name.c_str(), &interference_terms[i][j]);
			// 	}
			// }

			// for (int evt = 0; evt < n_events; ++evt)
			// {
			// 	for (int i = 0; i < npartials; ++i)
			// 	{
			// 		for (int j = 0; j < npartials; ++j)
			// 		{
			// 			interference_terms[i][j] = h_event_interference[evt * npartials * npartials + i * npartials + j] * normFactor;
			// 		}
			// 	}
			// 	phspTree->Fill();
			// }

			phspTree->Write();
			delete phspTree;
		}

		auto plotconfig = config_parser_.getPlotConfigs();
		std::vector<MassHistConfig> masshist;
		std::vector<AngleHistConfig> anglehist;
		std::vector<DalitzHistConfig> dalitzhist;
		int mass_hist_count = 0;
		int angle_hist_count = 0;
		int dalitz_hist_count = 0;
		for (const auto &histConfig : plotconfig)
		{
			// 输出histConfig内容以进行调试
			// std::cout << "PlotConfig type: " << histConfig.type << std::endl;
			if (histConfig.type == "mass")
			{
				std::vector<std::string> particles = histConfig.particles[0];
				int bins = histConfig.bins[0];
				std::vector<double> range = histConfig.ranges[0];
				std::vector<std::string> display = histConfig.display;

				std::string hist_name = "mass" + std::to_string(mass_hist_count++);
				for (const auto &p : particles)
				{
					hist_name += "_" + p;
				}
				std::cout << "Creating mass histogram: " << hist_name << std::endl;
				masshist.emplace_back(hist_name, "", particles, bins, range, display);
			}
			else if (histConfig.type == "cosbeta")
			{
				// 处理角度直方图配置
				std::vector<std::vector<std::string>> particles = histConfig.particles;
				int bins = histConfig.bins[0];
				std::vector<double> range = histConfig.ranges[0];
				std::vector<std::string> display = histConfig.display;
				std::string hist_name = "cosbeta" + std::to_string(angle_hist_count++);
				for (const auto &pvec : particles)
				{
					hist_name += "_";
					for (const auto &p : pvec)
					{
						hist_name += p;
					}
				}
				std::cout << "Creating angle histogram: " << hist_name << std::endl;
				anglehist.emplace_back(hist_name, "", particles, bins, range, display);
			}
			else if (histConfig.type == "dalitz")
			{
				std::vector<std::vector<std::string>> particles = histConfig.particles;
				std::vector<int> bins = histConfig.bins;
				std::vector<std::vector<double>> ranges = histConfig.ranges;
				std::vector<std::string> display = histConfig.display;
				std::string hist_name = "dalitz" + std::to_string(dalitz_hist_count++);
				for (const auto &pvec : particles)
				{
					hist_name += "_";
					for (const auto &p : pvec)
					{
						hist_name += p;
					}
				}
				std::cout << "Creating dalitz histogram: " << hist_name << std::endl;
				dalitzhist.emplace_back(hist_name, "", particles, bins, ranges, display);
			}
		}

		for (const auto &histConfig : masshist)
		{
			TDirectory *histDir = rootFile->mkdir(histConfig.name.c_str());
			histDir->cd();

			TObjString xlabel_obj(histConfig.tex[0].c_str());
			TObjString ylabel_obj(histConfig.tex[1].c_str());
			xlabel_obj.Write("xlabel");
			ylabel_obj.Write("ylabel");
		}

		for (const auto &histConfig : anglehist)
		{
			TDirectory *histDir = rootFile->mkdir(histConfig.name.c_str());
			histDir->cd();

			TObjString xlabel_obj(histConfig.tex[0].c_str());
			TObjString ylabel_obj(histConfig.tex[1].c_str());
			xlabel_obj.Write("xlabel");
			ylabel_obj.Write("ylabel");
		}

		for (const auto &histConfig : dalitzhist)
		{
			TDirectory *histDir = rootFile->mkdir(histConfig.name.c_str());
			histDir->cd();

			TObjString xlabel_obj(histConfig.tex[0].c_str());
			TObjString ylabel_obj(histConfig.tex[1].c_str());
			xlabel_obj.Write("xlabel");
			ylabel_obj.Write("ylabel");
		}

		// 四动量index
		std::map<std::string, int> particleToIndex;
		for (int i = 0; i < particles_.size(); ++i)
		{
			particleToIndex[particles_[i].name] = i;
		}

		// 计算并保存data直方图
		if (!Vp4_data_.empty())
		{
			std::vector<TH1F *> masshist_data;
			for (const auto &histConfig : masshist)
			{
				TH1F *hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
				masshist_data.emplace_back(hist);
			}

			LorentzVector *device_momenta = convertToLorentzVector(Vp4_data_, particleToIndex);
			CalculateMassHist(device_momenta, particleToIndex, masshist, nullptr, masshist_data, Vp4_data_.begin()->second.size(), particleToIndex.size());
			for (size_t i = 0; i < masshist.size(); ++i)
			{
				TDirectory *histDir = rootFile->GetDirectory(masshist[i].name.c_str());
				histDir->cd();
				masshist_data[i]->Write("hdata");
				delete masshist_data[i];
			}

			std::vector<TH1F *> anglehist_data;
			for (const auto &histConfig : anglehist)
			{
				TH1F *hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
				anglehist_data.emplace_back(hist);
			}
			CalculateAngleHist(device_momenta, particleToIndex, anglehist, nullptr, anglehist_data, Vp4_data_.begin()->second.size(), particleToIndex.size());
			for (size_t i = 0; i < anglehist.size(); ++i)
			{
				TDirectory *histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
				histDir->cd();
				anglehist_data[i]->Write("hdata");
				delete anglehist_data[i];
			}

			std::vector<TH2F *> dalitzhist_data;
			for (const auto &histConfig : dalitzhist)
			{
				TH2F *hist = new TH2F(histConfig.name.c_str(), histConfig.title.c_str(),
									  histConfig.bins[0], histConfig.range[0][0], histConfig.range[0][1],
									  histConfig.bins[1], histConfig.range[1][0], histConfig.range[1][1]);
				dalitzhist_data.emplace_back(hist);
			}
			CalculateDalitzHist(device_momenta, particleToIndex, dalitzhist, nullptr, dalitzhist_data, Vp4_data_.begin()->second.size(), particleToIndex.size());
			for (size_t i = 0; i < dalitzhist.size(); ++i)
			{
				TDirectory *histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
				histDir->cd();
				dalitzhist_data[i]->Write("hdata");
				delete dalitzhist_data[i];
			}

			// if (device_momenta != nullptr)
			cudaFree(device_momenta);
		}

		// 计算并保存拟合结果直方图
		if (!Vp4_phsp_.empty())
		{
			std::vector<TH1F *> masshist_fit;
			for (const auto &histConfig : masshist)
			{
				TH1F *hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
				masshist_fit.emplace_back(hist);
			}

			LorentzVector *phsp_momenta = convertToLorentzVector(Vp4_phsp_, particleToIndex);
			CalculateMassHist(phsp_momenta, particleToIndex, masshist, d_final_result, masshist_fit, Vp4_phsp_.begin()->second.size(), particleToIndex.size());
			// CalculateMassHist(phsp_momenta, particleToIndex, masshist, h_total_results, masshist_fit, Vp4_phsp_.begin()->second.size(), particleToIndex.size());
			for (size_t i = 0; i < masshist.size(); ++i)
			{
				TDirectory *histDir = rootFile->GetDirectory(masshist[i].name.c_str());
				histDir->cd();

				TH1F *hist = masshist_fit[i];
				hist->Scale(normFactor);
				hist->Write("hfit");
				delete masshist_fit[i];
			}

			std::vector<TH1F *> anglehist_fit;
			for (const auto &histConfig : anglehist)
			{
				TH1F *hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
				anglehist_fit.emplace_back(hist);
			}
			CalculateAngleHist(phsp_momenta, particleToIndex, anglehist, d_final_result, anglehist_fit, Vp4_phsp_.begin()->second.size(), particleToIndex.size());
			for (size_t i = 0; i < anglehist.size(); ++i)
			{
				TDirectory *histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
				histDir->cd();
				TH1F *hist = anglehist_fit[i];
				hist->Scale(normFactor);
				hist->Write("hfit");
				delete anglehist_fit[i];
			}

			std::vector<TH2F *> dalitzhist_fit;
			for (const auto &histConfig : dalitzhist)
			{
				TH2F *hist = new TH2F(histConfig.name.c_str(), histConfig.title.c_str(),
									  histConfig.bins[0], histConfig.range[0][0], histConfig.range[0][1],
									  histConfig.bins[1], histConfig.range[1][0], histConfig.range[1][1]);
				dalitzhist_fit.emplace_back(hist);
			}
			CalculateDalitzHist(phsp_momenta, particleToIndex, dalitzhist, d_final_result, dalitzhist_fit, Vp4_phsp_.begin()->second.size(), particleToIndex.size());
			for (size_t i = 0; i < dalitzhist.size(); ++i)
			{
				TDirectory *histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
				histDir->cd();
				TH2F *hist = dalitzhist_fit[i];
				hist->Scale(normFactor);
				hist->Write("hfit");
				delete dalitzhist_fit[i];
			}

			cudaFree(phsp_momenta);
		}

		// 计算并保存本底直方图
		if (!Vp4_bkg_.empty())
		{
			std::vector<TH1F *> masshist_bkg;
			for (const auto &histConfig : masshist)
			{
				TH1F *hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
				masshist_bkg.emplace_back(hist);
			}

			LorentzVector *device_momenta = convertToLorentzVector(Vp4_bkg_, particleToIndex);
			CalculateMassHist(device_momenta, particleToIndex, masshist, bkg_weights_, masshist_bkg, Vp4_bkg_.begin()->second.size(), particleToIndex.size());
			for (size_t i = 0; i < masshist.size(); ++i)
			{
				TDirectory *histDir = rootFile->GetDirectory(masshist[i].name.c_str());
				histDir->cd();
				masshist_bkg[i]->Write("hbkg");
				delete masshist_bkg[i];
			}

			std::vector<TH1F *> anglehist_bkg;
			for (const auto &histConfig : anglehist)
			{
				TH1F *hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
				anglehist_bkg.emplace_back(hist);
			}
			CalculateAngleHist(device_momenta, particleToIndex, anglehist, bkg_weights_, anglehist_bkg, Vp4_bkg_.begin()->second.size(), particleToIndex.size());
			for (size_t i = 0; i < anglehist.size(); ++i)
			{
				TDirectory *histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
				histDir->cd();
				anglehist_bkg[i]->Write("hbkg");
				delete anglehist_bkg[i];
			}

			std::vector<TH2F *> dalitzhist_bkg;
			for (const auto &histConfig : dalitzhist)
			{
				TH2F *hist = new TH2F(histConfig.name.c_str(), histConfig.title.c_str(),
									  histConfig.bins[0], histConfig.range[0][0], histConfig.range[0][1],
									  histConfig.bins[1], histConfig.range[1][0], histConfig.range[1][1]);
				dalitzhist_bkg.emplace_back(hist);
			}
			CalculateDalitzHist(device_momenta, particleToIndex, dalitzhist, bkg_weights_, dalitzhist_bkg, Vp4_bkg_.begin()->second.size(), particleToIndex.size());
			for (size_t i = 0; i < dalitzhist.size(); ++i)
			{
				TDirectory *histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
				histDir->cd();
				dalitzhist_bkg[i]->Write("hbkg");
				delete dalitzhist_bkg[i];
			}

			cudaFree(device_momenta);
		}

		//
		if (!Vp4_phsp_.empty())
		{
			LorentzVector *device_momenta = convertToLorentzVector(Vp4_phsp_, particleToIndex);
			for (int i = 0; i < npartials; ++i)
			{
				// 为当前部分创建直方图
				// std::vector<TH1F *> partialHists;
				std::vector<TH1F *> masshist_partial;
				for (const auto &histConfig : masshist)
				{
					TH1F *hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
					masshist_partial.push_back(hist);
				}

				CalculateMassHist(device_momenta, particleToIndex, masshist, &d_partial_result[i * n_events], masshist_partial, Vp4_phsp_.begin()->second.size(), particleToIndex.size());
				for (size_t j = 0; j < masshist_partial.size(); ++j)
				{

					TDirectory *histDir = rootFile->GetDirectory(masshist[j].name.c_str());
					histDir->cd();

					// std::string partial_dir_name = "h_" + amplitude_names_[i];
					std::string partial_dir_name = "h_" + resonance_names_[i];

					TH1F *hist = masshist_partial[j];
					hist->Scale(normFactor);
					hist->Write(partial_dir_name.c_str());

					delete hist;
				}
				masshist_partial.clear();

				std::vector<TH1F *> anglehist_partial;
				for (const auto &histConfig : anglehist)
				{
					TH1F *hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
					anglehist_partial.push_back(hist);
				}
				CalculateAngleHist(device_momenta, particleToIndex, anglehist, &d_partial_result[i * n_events], anglehist_partial, Vp4_phsp_.begin()->second.size(), particleToIndex.size());
				for (size_t j = 0; j < anglehist_partial.size(); ++j)
				{
					TDirectory *histDir = rootFile->GetDirectory(anglehist[j].name.c_str());
					histDir->cd();

					std::string partial_dir_name = "h_" + resonance_names_[i];

					TH1F *hist = anglehist_partial[j];
					hist->Scale(normFactor);
					hist->Write(partial_dir_name.c_str());
					delete anglehist_partial[j];
				}
				anglehist_partial.clear();
			}
			cudaFree(device_momenta);
		}

		// 关闭 ROOT 文件
		rootFile->Close();
		delete rootFile;

		// std::cout << "Data written to ROOT file: " << filename << std::endl;

		// 释放设备内存
		// cudaFree(d_row_results);
		cudaFree(d_final_result);
		cudaFree(d_partial_result);
		cudaFree(d_nSLvectors);
		cudaFree(d_total_integral);
		cudaFree(d_interference_matrix);
		// cudaFree(d_event_interference);
		delete[] h_total_results;
		delete[] h_partial_results;
	}

	torch::Tensor getDataTensor() const
	{
		torch::Tensor output = torch::from_blob(data_fix_, {data_length * n_gls_}, torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

		return output;
	}

	torch::Tensor getPhspTensor() const
	{
		torch::Tensor output = torch::from_blob(phsp_fix_, {phsp_length * n_gls_}, torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

		return output;
	}

	torch::Tensor getTruthTensor() const
	{
		const auto &data_files = config_parser_.getDataFiles();
		if (data_files.count("phsp_truth") > 0)
		{
			std::vector<std::string> particles_names;
			for (const auto &particle : particles_)
			{
				particles_names.push_back(particle.name);
			}

			// std::cout << "Reading phase space truth samples..." << std::endl;
			std::map<std::string, std::vector<LorentzVector>> Vp4_truth = readMomentaFromDat(data_files.at("phsp_truth"), config_parser_.getDataOrder(), particles_names);
			std::cout << "Phase space truth events: " << Vp4_truth.begin()->second.size() << std::endl;
			// std::cout << "Calculating phase space truth amplitudes..." << std::endl;
			cuComplex *truth_fix = calculateAmplitudes(Vp4_truth);
			int truth_length = Vp4_truth.begin()->second.size() * n_polar_;

			torch::Tensor output = torch::from_blob(truth_fix, {truth_length * n_gls_}, torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

			cudaFree(truth_fix);

			return output;
		}
		else
		{
			std::cerr << "No phsp_truth data file specified in the configuration." << std::endl;
			return torch::empty({0}, torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA));
		}
	}

	torch::Tensor getBkgTensor() const
	{
		torch::Tensor output = torch::from_blob(bkg_fix_, {bkg_length * n_gls_}, torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

		return output;
	}

	torch::Tensor getBkgWeightsTensor() const
	{
		if (bkg_weights_ != nullptr && bkg_length > 0)
		{
			torch::Tensor output = torch::from_blob(bkg_weights_, {bkg_length}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)).clone();
			return output;
		}
		else
		{
			return torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
		}
	}

	std::vector<std::vector<int>> getConstraintsIndex() const
	{
		return con_trans_id_;
	}

	std::vector<std::vector<std::pair<double, double>>> getConstraintsValues() const
	{
		std::vector<std::vector<std::pair<double, double>>> output;
		for (const auto &vec : con_trans_values_)
		{
			std::vector<std::pair<double, double>> temp;
			for (const auto &val : vec)
			{
				temp.emplace_back(std::make_pair(real(val), imag(val)));
			}
			output.push_back(temp);
		}
		return output;
	}

	int getNPolarizations() const
	{
		return n_polar_;
	}

	std::vector<std::string> getAmplitudeNames() const
	{
		return amplitude_names_;
	}

private:
	int n_gls_;
	int n_polar_ = 1;
	std::vector<int> nSLvectors_;
	cuComplex *data_fix_;
	int data_length;
	cuComplex *phsp_fix_;
	int phsp_length;
	cuComplex *bkg_fix_;
	double *bkg_weights_ = nullptr;
	int bkg_length;
	// cuComplex *truth_fix_;
	// int truth_length;

	// 四动量数据
	std::map<std::string, std::vector<LorentzVector>> Vp4_data_;
	std::map<std::string, std::vector<LorentzVector>> Vp4_phsp_;
	std::map<std::string, std::vector<LorentzVector>> Vp4_bkg_;
	// std::map<std::string, std::vector<LorentzVector>> Vp4_truth_;

	// amplitude 信息
	std::vector<std::string> amplitude_names_;
	std::vector<std::string> resonance_names_;
	std::vector<std::string> legends_;

	// 约束信息
	// std::vector<std::pair<int, int>> conjugate_pairs_;
	std::vector<std::vector<int>> con_trans_id_;
	std::vector<std::vector<std::complex<double>>> con_trans_values_;

	// config 初始化
	ConfigParser config_parser_;
	std::vector<Particle> particles_;
	std::unordered_map<std::string, Resonance> resonances_;
	int n_amplitudes_ = 0;
	std::vector<ChainInfo> chains_info_;

	void initialize(std::string config_file = "config.yml")
	{
		// 读取配置文件
		// std::string config_file = "config.yml";
		// config_parser_(config_file);
		std::cout << "Reading config file: " << config_file << std::endl;
		std::cout << "Particles: " << config_parser_.getParticles().size() << std::endl;
		std::cout << "Decay chains: " << config_parser_.getDecayChains().size() << std::endl;
		std::cout << "Resonances: " << config_parser_.getResonances().size() << std::endl;
		std::cout << "Constraints: " << config_parser_.getConstraints().size() << std::endl;

		// 初始化粒子信息
		initializeParticles();
		// 初始化极化状态
		initializePolarization();
		// 初始化衰变链
		initializeDecayChains();

		// // 获取配置信息
		// const auto &config_parser = calculator.getConfigParser();
		const auto &data_files = config_parser_.getDataFiles();
		const auto &data_order = config_parser_.getDataOrder();

		legends_ = config_parser_.getLegends();
		n_gls_ = n_amplitudes_;

		std::vector<std::string> particles_names;
		for (const auto &particle : particles_)
		{
			particles_names.push_back(particle.name);
		}

		// 计算相空间振幅
		std::cout << "Reading phase space samples..." << std::endl;
		Vp4_phsp_ = readMomentaFromDat(data_files.at("phsp"), data_order, particles_names);
		std::cout << "Phase space events: " << Vp4_phsp_.begin()->second.size() << std::endl;
		std::cout << "Calculating phase space amplitudes..." << std::endl;
		phsp_fix_ = calculateAmplitudes(Vp4_phsp_);
		phsp_length = Vp4_phsp_.begin()->second.size() * n_polar_;

		// 计算数据振幅
		std::cout << "Reading data samples..." << std::endl;
		Vp4_data_ = readMomentaFromDat(data_files.at("data"), data_order, particles_names);
		std::cout << "data events: " << Vp4_data_.begin()->second.size() << std::endl;
		std::cout << "Calculating data amplitudes..." << std::endl;
		data_fix_ = calculateAmplitudes(Vp4_data_);
		data_length = Vp4_data_.begin()->second.size() * n_polar_;

		// 计算本底振幅
		if (data_files.count("bkg") > 0)
		{
			std::cout << "Reading background samples..." << std::endl;
			Vp4_bkg_ = readMomentaFromDat(data_files.at("bkg"), data_order, particles_names);
			std::cout << "Background events: " << Vp4_bkg_.begin()->second.size() << std::endl;
			std::cout << "Calculating background amplitudes..." << std::endl;
			bkg_fix_ = calculateAmplitudes(Vp4_bkg_);
			bkg_length = Vp4_bkg_.begin()->second.size() * n_polar_;

			if (data_files.count("bkg_weights") > 0)
			{
				bkg_weights_ = readWeightsFromFile(data_files.at("bkg_weights"), Vp4_bkg_.begin()->second.size());
			}
		}

		// 计算分支比
		// if (data_files.count("phsp_truth") > 0)
		// {
		// 	std::map<std::string, std::vector<LorentzVector>> Vp4_truth = readMomentaFromDat(data_files.at("phsp_truth"), config_parser_.getDataOrder(), particles_names);
		// 	truth_fix_ = calculateAmplitudes(Vp4_truth);
		// 	truth_length = Vp4_truth.begin()->second.size() * n_polar_;
		// }

		NLLFunction::setConstraints(con_trans_id_, con_trans_values_);

		std::cout << "Number of partial waves (n_gls_): " << n_gls_ << std::endl;
		std::cout << "Number of amplitude names: " << amplitude_names_.size() << std::endl;
		// std::cout << "Number of constraints: " << conjugate_pairs_.size() << std::endl;
		std::cout << "Initialization complete." << std::endl;
	}

	void initializeParticles()
	{
		const auto &config_particles = config_parser_.getParticles();
		for (const auto &particle_config : config_particles)
		{
			particles_.push_back(particle_config);
		}
	}

	void initializePolarization()
	{
		n_polar_ = 1;

		for (const auto &particle : particles_)
		{
			if (particle.spin > 0)
			{
				// n_polar_ *= (2 * particle.spin + 1);
				n_polar_ *= particle.spin;
			}
		}

		std::cout << "polarization: " << n_polar_ << std::endl;
	}

	void initializeDecayChains()
	{
		auto chains = config_parser_.getDecayChains();

		const auto &config_resonances = config_parser_.getResonances();

		// 获取总振幅长度
		for (const auto &chain : chains)
		{
			// std::cout << "Processing decay chain: " << chain.name << std::endl;

			ChainInfo chain_info;
			chain_info.name = chain.name;

			std::map<std::pair<std::string, std::vector<int>>, std::vector<Resonance>> intermediate_resonance_map;
			std::vector<std::vector<Particle>> intermediate_particles;
			for (const auto &res_chain : chain.resonance_chains)
			{
				// std::cout << "  Intermediate: " << res_chain.intermediate << std::endl;
				std::vector<Particle> particles;
				for (const auto &spin_chain : res_chain.spin_chains)
				{
					// std::cout << "    Spin-Parity options: ";

					Particle intermediate_particle = {res_chain.intermediate, static_cast<int>(spin_chain.spin_parity[0]), static_cast<int>(spin_chain.spin_parity[1]), -1};

					std::vector<Resonance> resonance_list;
					// std::cout << " Resonance: " << std::endl;
					for (const auto &resonance_name : spin_chain.resonances)
					{
						// std::cout << resonance_name << std::endl;

						for (const auto &[name, res_config] : config_resonances)
						{
							if (name == resonance_name)
							{
								if (res_config.J == spin_chain.spin_parity[0] && spin_chain.spin_parity[1])
								{
									resonance_list.emplace_back(name, res_chain.intermediate, intermediate_particle.spin, intermediate_particle.parity, res_config.type, res_config.parameters);
									// std::cout << "      Added resonance: " << name << " J: " << res_config.J << " P: " << res_config.P << std::endl;
								}
								else
								{
									std::cout << "      Skipped resonance (J,P mismatch): " << name << " J: " << res_config.J << " P: " << res_config.P << std::endl;
								}
							}
						}
					}
					// std::cout << std::endl;

					std::pair<std::string, std::vector<int>> key = {res_chain.intermediate, {spin_chain.spin_parity[0], spin_chain.spin_parity[1]}};
					intermediate_resonance_map[key] = resonance_list;
					particles.push_back(intermediate_particle);
				}
				intermediate_particles.push_back(particles);
			}

			chain_info.intermediate_resonance_map = intermediate_resonance_map;

			std::vector<std::vector<Particle>> intermediate_combs = {{}};
			for (const auto &particleList : intermediate_particles)
			{
				std::vector<std::vector<Particle>> temp;
				for (const auto &comb : intermediate_combs)
				{
					for (const auto &particle : particleList)
					{
						std::vector<Particle> new_res = comb;
						// std::cout << " Adding particle to combination: " << particle.name << " J: " << particle.spin << " P: " << particle.parity << std::endl;
						new_res.push_back(particle);
						temp.push_back(new_res);
					}
				}
				intermediate_combs = std::move(temp);
			}

			chain_info.intermediate_combs = intermediate_combs;

			for (auto comb : intermediate_combs)
			{
				AmpCasDecay cas(particles_);
				for (const auto &step : chain.decay_steps)
				{
					std::array<int, 3> spins = {0};
					std::array<int, 3> parities = {0};
					for (auto particle : particles_)
					{
						if (particle.name == step.mother)
						{
							// std::cout << "mother: " << particle.name << " " << particle.spin << " " << particle.parity << std::endl;
							spins[0] = particle.spin;
							parities[0] = particle.parity;
						}

						for (int i = 0; i < step.daughters.size(); i++)
						{
							if (particle.name == step.daughters[i])
							{
								// std::cout << "daugters: " << particle.name << " " << particle.spin << " " << particle.parity << std::endl;
								spins[i + 1] = particle.spin;
								parities[i + 1] = particle.parity;
							}
						}
					}
					for (auto res_jp : comb)
					{
						if (res_jp.name == step.mother)
						{
							// std::cout << "mother: " << res_jp.name << " " << res_jp.spin << std::endl;
							spins[0] = res_jp.spin;
							parities[0] = res_jp.parity;
						}

						for (int i = 0; i < step.daughters.size(); i++)
						{
							if (res_jp.name == step.daughters[i])
							{
								// std::cout << "daugters: " << res_jp.name << " " << res_jp.spin << " " << res_jp.parity << std::endl;
								spins[i + 1] = res_jp.spin;
								parities[i + 1] = res_jp.parity;
							}
						}
					}
					cas.addDecay(Amp2BD(spins, parities), step.mother, step.daughters[0], step.daughters[1]);

					// 输出decay chain结构
					std::cout << step.mother << "(";
					if (spins[0] % 2 != 0)
						std::cout << (spins[0] - 1) / 2;
					else
						std::cout << spins[0] - 1 << "/2";
					if (parities[0] == 1)
						std::cout << "+)";
					else if (parities[0] == -1)
						std::cout << "-)";
					std::cout << "->";
					for (int i = 0; i < step.daughters.size(); i++)
					{
						std::cout << step.daughters[i] << "(";
						if (spins[i + 1] % 2 != 0)
							std::cout << (spins[i + 1] - 1) / 2;
						else
							std::cout << spins[i + 1] - 1 << "/2";
						if (parities[i + 1] == 1)
							std::cout << "+)";
						else if (parities[i + 1] == -1)
							std::cout << "-)";
					}
					std::cout << ", ";
				}

				auto slcombs = cas.getSLCombinations();
				std::cout << "SL:";
				for (auto slcomb : slcombs)
				{
					std::cout << "{";
					for (auto sl : slcomb)
					{
						std::cout << "(" << (sl.S - 1) / 2.0 << ", " << sl.L << ")";
					}
					std::cout << "}";
				}
				std::cout << std::endl;

				std::vector<std::vector<std::pair<std::string, std::string>>> resonance_combinations = {{}};
				for (const auto &particle : comb)
				{
					std::pair<std::string, std::vector<int>> key = {particle.name, {particle.spin, particle.parity}};
					const auto &resonance_list = intermediate_resonance_map[key];

					std::vector<std::vector<std::pair<std::string, std::string>>> temp;
					for (const auto &res_comb : resonance_combinations)
					{
						for (const auto &resonance : resonance_list)
						{
							std::vector<std::pair<std::string, std::string>> new_res_comb = res_comb;
							new_res_comb.push_back({resonance.getTag(), resonance.getName()});
							temp.push_back(new_res_comb);
						}
					}
					resonance_combinations = std::move(temp);
				}

				std::cout << "Resonance: ";
				for (size_t k = 0; k < resonance_combinations.size(); ++k)
				{
					n_amplitudes_ += slcombs.size();
					nSLvectors_.push_back(slcombs.size());

					std::string res_name = chain.name;
					std::cout << "{ ";
					for (size_t j = 0; j < resonance_combinations[k].size(); ++j)
					{
						const auto &res_pair = resonance_combinations[k][j];
						res_name += "-" + res_pair.first + "-" + res_pair.second;

						std::cout << res_pair.second; // 共振态名称
						if (j < resonance_combinations[k].size() - 1)
							std::cout << ", ";
					}
					if (k < resonance_combinations.size() - 1)
						std::cout << " }, ";
					else
						std::cout << "}";

					for (const auto &slcomb : slcombs)
					{
						std::string full_name = res_name + "-" + "SL";
						for (const auto &sl : slcomb)
						{
							full_name += "_" + std::to_string(sl.S) + std::to_string(sl.L);
						}
						amplitude_names_.push_back(full_name);
					}
					resonance_names_.push_back(res_name);
				}
				std::cout << std::endl;
			}

			chains_info_.push_back(chain_info);
		}

		// 设置约束条件
		auto constraints = config_parser_.getConstraints();

		for (const auto &constraint : constraints)
		{
			std::vector<std::vector<int>> amp_ids_con;

			for (const auto &amp_name : constraint.names)
			{
				std::vector<int> amp_ids;
				for (int i = 0; i < amplitude_names_.size(); ++i)
				{
					if (amplitude_names_[i].find(amp_name) != std::string::npos)
					{
						amp_ids.push_back(i);
					}
				}
				amp_ids_con.push_back(amp_ids);
			}

			// 生成所有组合
			// std::vector<std::vector<int>> all_combinations;
			// std::vector<std::vector<std::complex<double>>> con_values;
			int num_constraints = amp_ids_con.size();
			for (int i = 0; i < amp_ids_con[0].size(); ++i)
			{
				// // 先输出amp_ids_con内容和amplitude_names_对应关系，便于调试
				// std::cout << "Constraint: " << i;
				// for (int j = 0; j < num_constraints; ++j)
				// {
				// 	std::cout << amplitude_names_[amp_ids_con[j][i]] << "( " << amp_ids_con[j][i] << " ) ";
				// }
				// std::cout << std::endl;
				std::vector<int> combination;
				for (int j = 0; j < num_constraints; ++j)
				{
					combination.push_back(amp_ids_con[j][i]);
				}
				// all_combinations.push_back(combination);
				con_trans_id_.push_back(combination);

				// 第一个是{1+1j}, 后面是constraint.values
				std::vector<std::complex<double>> values = {std::complex<double>(1.0, 1.0)};
				for (const auto &val : constraint.values)
				{
					values.push_back(val);
				}
				// con_values.push_back(values);
				con_trans_values_.push_back(values);
			}

			// for (int i = 0; i < con_trans_id_.size(); ++i)
			// {
			// 	std::cout << "Constraint IDs: ";
			// 	for (const auto &id : con_trans_id_[i])
			// 	{
			// 		std::cout << id << " ";
			// 	}
			// 	std::cout << std::endl;
			// }
		}
	}

	cuComplex *calculateAmplitudes(const std::map<std::string, std::vector<LorentzVector>> &Vp4) const
	{
		cuComplex *d_all_amplitudes = nullptr;
		const size_t n_events = Vp4.begin()->second.size();
		cudaMalloc(&d_all_amplitudes, n_events * n_amplitudes_ * n_polar_ * sizeof(cuComplex));

		// std::cout << "Calculating amplitudes for " << n_events << " events..." << std::endl;
		// std::cout << "polarization states: " << n_polar_ << std::endl;
		// std::cout << "total amplitudes: " << n_amplitudes_ << std::endl;

		auto chains = config_parser_.getDecayChains();
		int gls_index = 0;
		for (auto chain : chains)
		{
			ChainInfo chain_info;
			for (auto chaininfo : chains_info_)
			{
				if (chaininfo.name == chain.name)
				{
					chain_info = chaininfo;
					break;
				}
			}
			auto intermediate_resonance_map = chain_info.intermediate_resonance_map;
			auto intermediate_combs = chain_info.intermediate_combs;

			for (auto comb : intermediate_combs)
			{
				AmpCasDecay cas(particles_);
				for (const auto &step : chain.decay_steps)
				{
					std::array<int, 3> spins = {0};
					std::array<int, 3> parities = {0};
					for (auto particle : particles_)
					{
						if (particle.name == step.mother)
						{
							spins[0] = particle.spin;
							parities[0] = particle.parity;
						}

						for (int i = 0; i < step.daughters.size(); i++)
						{
							if (particle.name == step.daughters[i])
							{
								spins[i + 1] = particle.spin;
								parities[i + 1] = particle.parity;
							}
						}
					}
					for (auto res_jp : comb)
					{
						if (res_jp.name == step.mother)
						{
							spins[0] = res_jp.spin;
							parities[0] = res_jp.parity;
						}

						for (int i = 0; i < step.daughters.size(); i++)
						{
							if (res_jp.name == step.daughters[i])
							{
								spins[i + 1] = res_jp.spin;
								parities[i + 1] = res_jp.parity;
							}
						}
					}
					cas.addDecay(Amp2BD(spins, parities), step.mother, step.daughters[0], step.daughters[1]);
				}

				auto slcombs = cas.getSLCombinations();

				std::vector<std::vector<Resonance>> resonance_combinations = {{}};
				for (const auto &particle : comb)
				{
					std::pair<std::string, std::vector<int>> key = {particle.name, {particle.spin, particle.parity}};
					std::vector<Resonance> &resonance_list = intermediate_resonance_map[key];

					std::vector<std::vector<Resonance>> temp;
					for (const auto &res_comb : resonance_combinations)
					{
						for (const auto &resonance : resonance_list)
						{
							std::vector<Resonance> new_res_comb = res_comb;
							new_res_comb.push_back(resonance); // 直接添加 Resonance 对象
							temp.push_back(std::move(new_res_comb));
						}
					}
					resonance_combinations = std::move(temp);
				}

				cas.computeSLAmps(Vp4);
				int nSLcombs = cas.getNSLCombs();
				int nEvents = cas.getNEvents();

				for (const auto resonance : resonance_combinations)
				{
					cas.getAmps(d_all_amplitudes, resonance, gls_index);
					gls_index += 1;
				}
			}
		}

		return d_all_amplitudes;
	}
};

// 定义Python模块
PYBIND11_MODULE(ctpwa, m)
{
	m.doc() = "ctpwa";
	// pybind11::class_<std::pair<int, int>>(m, "ConjugatePair")
	// 	.def_readonly("first", &std::pair<int, int>::first)
	// 	.def_readonly("second", &std::pair<int, int>::second);

	pybind11::class_<analysis>(m, "analysis")
		.def(pybind11::init<>())
		.def("getNLL", &analysis::getNLL)
		.def("getNVector", &analysis::getNVector)
		.def("getSLVectors", &analysis::getSLVectors)
		.def("writeResult", &analysis::writeResult)
		// .def("writeInterference", &analysis::writeInterference)
		.def("getDataTensor", &analysis::getDataTensor)
		.def("getPhspTensor", &analysis::getPhspTensor)
		.def("getTruthTensor", &analysis::getTruthTensor)
		.def("getBkgTensor", &analysis::getBkgTensor)
		.def("getBkgWeightsTensor", &analysis::getBkgWeightsTensor)
		.def("getConstraintsIndex", &analysis::getConstraintsIndex)
		.def("getConstraintsValues", &analysis::getConstraintsValues)
		.def("getAmplitudeNames", &analysis::getAmplitudeNames)
		.def("getNPolarizations", &analysis::getNPolarizations);
}
