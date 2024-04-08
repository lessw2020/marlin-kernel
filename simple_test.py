import torch
import marlin

def make_tensor(M, N, dtype):
    torch.cuda.manual_seed(3227)
    if dtype == torch.int32:
        # Fill with random integers for int32 type
        res = torch.randint(low=-2**31, high=2**31, size=(M, N), dtype=dtype, device="cuda")
    else:
        # Fill with normally distributed random values for other types
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    return res

if __name__ == '__main__':

    m = 16
    k = 4096
    n = 4096
    groupsize = 128
    g = k // groupsize

    a = make_tensor(m, k, dtype=torch.float16)
    b = make_tensor(k//8, n, dtype=torch.int32)
    c = make_tensor(m, n, dtype=torch.float16)
    workspace = torch.zeros(n//128*16, device="cuda")

    zeros = make_tensor(g, n//8, torch.int32)
    scales = make_tensor(g, n, torch.float16)



    c = torch.empty((m,n), dtype=torch.float16, device = a.device)
    marlin.mul(a, b, c, scales, workspace, sms=132)

    print(c[0:10])
