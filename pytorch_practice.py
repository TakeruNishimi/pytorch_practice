import torch


def main():
    t1 = torch.tensor([[1, 2, 5],
                       [3, 4, 7]], dtype=torch.float)
    # どれかでも一つがfloatならすべてfloat

    print(t1)
    print(t1.dtype)
    print(t1.size())
    print(t1.shape)

    t2 = torch.ones(size=(2, 3))
    print(t2)
    print(t2.dtype)
    print(t2.size())

    # int + floatやサイズが違うとError
    print(t1 + t2)


if __name__ == '__main__':
    main()
