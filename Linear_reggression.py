import torch


def main():
    torch.manual_seed(0)

    #     真の重み
    w_true = torch.tensor([1, 2, 3], dtype=float)

    N = 100
    X = torch.cat([torch.ones(N, 1),
                   torch.randn(N, 2)], dim=1)
    print(X)
    print(X.size())


if __name__ == '__main__':
    main()
