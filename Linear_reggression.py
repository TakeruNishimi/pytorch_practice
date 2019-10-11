import matplotlib.pyplot as plt
import torch


# methodsを使ってリファクタリング

def prepare_data(N: int, w_true: torch.Tensor):
    X = torch.cat([torch.ones(N, 1),
                   torch.randn(N, 2)],
                  dim=1)
    # 1列目が1なのは内積の計算のため　x0=1
    # print(X)
    # print(X.size())
    noise = torch.randn(N) * 0.5
    y = torch.mv(X, w_true) + noise
    # mv 内積

    return X, y


def main():
    torch.manual_seed(0)

    #     真の重み
    w_true = torch.tensor([1., 2., 3.])

    N = 100
    X, y = prepare_data(N, w_true)

    # requires_grad 計算グラフの保持　→　逆伝播するため
    w = torch.randn(w_true.size(0), requires_grad=True)

    # 学習におけるハイパーパラメータ
    learning_rate = 0.01
    num_epochs = 200
    loss_list = []
    # 表示のために 1からスタート
    for epoch in range(1, num_epochs + 1):
        # 前エポックの勾配をリセット
        w.grad = None

        # 予測の出力、計算
        y_pred = torch.mv(X, w)
        # 損失の計算
        loss = torch.mean((y_pred - y) ** 2)
        # 誤差逆伝播による勾配計算
        # フォアードプロパゲーションのあとバックプロパゲーションで微分
        loss.backward()

        print(f'Epoch = {epoch}: Loss= {loss.item():.4f} w={w} dL/dw = {w.grad.data}')
        # 重みの更新
        w.data = w - learning_rate * w.grad.data
        loss_list.append(loss.item())
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    main()
