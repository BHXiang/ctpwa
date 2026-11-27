import json, sys, io, contextlib
import torch, ctpwa
cfg, out = sys.argv[1], sys.argv[2]
with contextlib.redirect_stdout(io.StringIO()):
    ana = ctpwa.analysis(cfg)
dev = torch.device("cuda:0") if ctpwa.backend() == "cuda" else torch.device("cpu")
n = ana.getNVector(); nt = ana.getNFreeTheta()
p = torch.zeros(2*n+nt, dtype=torch.float64)
p[0] = 1.0
g = torch.Generator(device="cpu").manual_seed(20240917)
p[1:n] = torch.rand(n-1, generator=g) * 0.5
p[n+1:2*n] = (torch.rand(n-1, generator=g) - 0.5) * 0.5
if nt: p[2*n:] = ana.getFreeResParams()[0].to(torch.float64)
p = p.to(dev)
with contextlib.redirect_stdout(io.StringIO()):
    nll = float(ana.getNLL(p).item())
    H = ana.getHessian(p.detach()).cpu().double().reshape(-1).tolist()
json.dump({"backend": ctpwa.backend(), "nll": nll, "hess": H}, open(out, "w"))
print("ok", ctpwa.backend(), len(H))
