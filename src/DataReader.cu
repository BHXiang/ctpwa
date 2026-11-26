// src/DataReader.cu — 输入数据读取（纯 host；见 include/DataReader.h）
#include <DataReader.h>

#include <TFile.h>
#include <TLorentzVector.h>
#include <TTree.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

std::map<std::string, std::vector<LorentzVector>> readMomentaFromDat(
    const std::vector<std::string>& fileinfo,
    const std::vector<std::string>& particleNames,
    const std::vector<std::string>& particlelists,
    int nEvents,
    int offset)
{
    std::map<std::string, std::vector<LorentzVector>> fullMomenta;
    for (const auto& name : particlelists)
    {
        fullMomenta[name] = std::vector<LorentzVector>();
    }

    // 检查输入参数
    if (fileinfo.size() < 2)
    {
        std::cerr
            << "Error: fileinfo must contain at least file type and filename"
            << std::endl;
        return fullMomenta;
    }

    std::string fileType = fileinfo[0];
    std::string filename = fileinfo[1];

    std::unordered_set<std::string> particleNameSet(particleNames.begin(),
        particleNames.end());
    std::string initialName;
    bool foundParticle = false;

    for (const auto& name : particlelists)
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
                std::cerr
                    << "Error: Found multiple particles in particlelists not "
                    "present in particleNames"
                    << std::endl;
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
        int skippedEvents = 0;
        int particlesPerEvent = particleNames.size();
        int skipLines = offset * particlesPerEvent;

        while (std::getline(file, line))
        {
            if (line.empty())
                continue;

            std::istringstream iss(line);
            double E, px, py, pz;

            if (iss >> E >> px >> py >> pz)
            {
                // Skip lines for offset events
                if (lineCount < skipLines) {
                    lineCount++;
                    continue;
                }

                // 根据行号确定粒子类型
                int particleIndex = (lineCount - skipLines) % particlesPerEvent;
                const std::string& particleName = particleNames[particleIndex];

                fullMomenta[particleName].emplace_back(E, px, py, pz);
                lineCount++;

                // 每读完一组粒子表示完成一个事件
                if (particleIndex == particlesPerEvent - 1)
                {
                    LorentzVector initialMomentum(0, 0, 0, 0);
                    for (const auto& name : particleNames)
                    {
                        initialMomentum =
                            initialMomentum + fullMomenta[name].back();
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
                std::cerr << "Warning: Invalid line format: " << line
                    << std::endl;
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
            std::cerr
                << "Error: For ROOT files, fileinfo must contain at least file "
                "type, filename and TTree name"
                << std::endl;
            return fullMomenta;
        }

        std::string treeName = fileinfo[2];

        // 打开ROOT文件
        TFile* file = TFile::Open(filename.c_str(), "READ");
        if (!file || file->IsZombie())
        {
            std::cerr << "Error: Cannot open ROOT file " << filename
                << std::endl;
            return fullMomenta;
        }

        // 获取TTree
        TTree* tree = (TTree*)file->Get(treeName.c_str());
        if (!tree)
        {
            std::cerr << "Error: Cannot find TTree " << treeName << " in file "
                << filename << std::endl;
            file->Close();
            delete file;
            return fullMomenta;
        }

        // 准备读取TLorentzVector的分支
        std::vector<TLorentzVector*> particleLV(particleNames.size());
        std::vector<std::string> branchNames;

        // 如果提供了分支名，使用提供的分支名
        if (fileinfo.size() >= 2 + particleNames.size())
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

        // 读取事件（支持 offset + nEvents 分块: truth/phsp 积分按批读取）
        const Long64_t nTotal = tree->GetEntries();
        const Long64_t iBegin = std::min<Long64_t>(std::max<Long64_t>(offset, 0), nTotal);
        Long64_t iEnd = nTotal;
        if (nEvents > 0)
            iEnd = std::min<Long64_t>(iBegin + static_cast<Long64_t>(nEvents), nTotal);

        for (Long64_t iEvent = iBegin; iEvent < iEnd; ++iEvent)
        {
            tree->GetEntry(iEvent);

            // 读取每个粒子的四动量
            for (size_t i = 0; i < particleNames.size(); ++i)
            {
                const std::string& particleName = particleNames[i];
                TLorentzVector* lv = particleLV[i];

                // 转换为你的LorentzVector类型
                // 假设你的LorentzVector构造函数接受(E, px, py, pz)
                fullMomenta[particleName].emplace_back(lv->E(), lv->Px(),
                    lv->Py(), lv->Pz());
            }

            // 计算初始粒子的四动量
            LorentzVector initialMomentum(0, 0, 0, 0);
            for (const auto& name : particleNames)
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
        // std::cerr << "Error: ROOT support not compiled in. Please define
        // USE_ROOT and link with ROOT libraries." << std::endl; return
        // fullMomenta; #endif
    }
    else
    {
        std::cerr << "Error: Unknown file type: " << fileType << std::endl;
        return fullMomenta;
    }

    return fullMomenta;
}
std::vector<double> readWeightValuesFromFile(
    const std::vector<std::string>& fileinfo, long expected)
{
    std::vector<double> weights;

    if (fileinfo.empty()) {
        std::cerr << "Error: fileinfo is empty" << std::endl;
        return weights;
    }

    const std::string fileType = fileinfo[0];
    const std::string filename = fileinfo.size() > 1 ? fileinfo[1] : "";

    // 常量权重: 只传一个值，所有事例乘以这个 weight
    if (fileinfo.size() == 1) {
        if (expected < 0) {
            std::cerr << "Error: constant weight needs the event count" << std::endl;
            return weights;
        }
        double const_weight = 0.0;
        try {
            const_weight = std::stod(fileType);
        } catch (const std::exception&) {
            std::cerr << "Error: cannot parse constant weight '" << fileType << "'" << std::endl;
            return weights;
        }
        weights.assign(static_cast<size_t>(expected), const_weight);
        return weights;
    }

    // DAT 文件: 每行一个权重
    if (fileType == "dat")
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return weights;
        }

        double weight = 0.0;
        while (file >> weight)
        {
            weights.push_back(weight);
        }

        file.close();
    }
    // ROOT 文件: 从 TTree 的指定分支逐事件读
    else if (fileType == "ROOT" || fileType == "root")
    {
        if (fileinfo.size() < 4)
        {
            std::cerr << "Error: For ROOT files, fileinfo must contain at least file "
                "type, filename, TTree name and weight branch name" << std::endl;
            return weights;
        }

        const std::string treeName = fileinfo[2];
        const std::string branchName = fileinfo[3];

        TFile* file = TFile::Open(filename.c_str(), "READ");
        if (!file || file->IsZombie())
        {
            std::cerr << "Error: Cannot open ROOT file " << filename << std::endl;
            return weights;
        }

        TTree* tree = (TTree*)file->Get(treeName.c_str());
        if (!tree)
        {
            std::cerr << "Error: Cannot find TTree " << treeName << " in file " << filename << std::endl;
            file->Close();
            delete file;
            return weights;
        }

        double weight = 0.0;
        tree->SetBranchAddress(branchName.c_str(), &weight);

        const Long64_t nEntries = tree->GetEntries();
        weights.reserve(static_cast<size_t>(nEntries));
        for (Long64_t iEvent = 0; iEvent < nEntries; ++iEvent)
        {
            tree->GetEntry(iEvent);
            weights.push_back(weight);
        }

        file->Close();
        delete file;
    }
    else
    {
        std::cerr << "Error: Unknown file type: " << fileType << std::endl;
    }

    return weights;
}