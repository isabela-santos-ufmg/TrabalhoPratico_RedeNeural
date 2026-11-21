import csv
import itertools
import os
import time
import numpy as np
import learn

def run_grid_search():
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    epochs_list = [50, 100, 200, 500]
    hidden_sizes = [4, 16, 64, 256]

    os.makedirs("resultados/curvas", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    print("Carregando dataset MNIST...")
    X_train, X_test, y_train, y_test = learn.load_mnist()
    input_size = X_train.shape[1]   # 784
    output_size = y_train.shape[1]  # 10

    summary_path = "resultados/resultados_64_experimentos.csv"
    combinacoes = list(itertools.product(learning_rates, epochs_list, hidden_sizes))
    total = len(combinacoes)

    melhor_acc = -1.0
    melhor_config = None
    melhores_params = None
    melhores_losses_100 = None
    melhores_accs_100 = None

    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experimento",
            "learning_rate",
            "epochs",
            "hidden_size",
            "final_train_loss",
            "final_test_accuracy",
            "tempo_segundos"
        ])

        for idx, (lr, max_epochs, hidden_size) in enumerate(combinacoes, start=1):
            inicio = time.time()

            np.random.seed(42)
            parameters = learn.initialize_parameters(input_size, hidden_size, output_size)

            losses_all = []
            accs_all = []
            losses_100 = []
            accs_100 = []

            for epoch in range(max_epochs):
                parameters, train_loss = learn.gradient_descent_step(
                    X_train, y_train, parameters, learning_rate=lr
                )
                y_hat = learn.forward(X_test, parameters)[-1]
                test_acc = learn.accuracy(y_test, y_hat)

                losses_all.append(float(train_loss))
                accs_all.append(float(test_acc))

                if epoch % 100 == 0:
                    losses_100.append(float(train_loss))
                    accs_100.append(float(test_acc))

            duracao = time.time() - inicio
            final_loss = losses_all[-1]
            final_acc = accs_all[-1]

            writer.writerow([
                idx,
                lr,
                max_epochs,
                hidden_size,
                f"{final_loss:.6f}",
                f"{final_acc:.6f}",
                f"{duracao:.2f}"
            ])
            f.flush()

            curve_path = f"resultados/curvas/lr_{lr}_ep_{max_epochs}_h_{hidden_size}.csv"
            with open(curve_path, mode="w", newline="", encoding="utf-8") as cf:
                cwriter = csv.writer(cf)
                cwriter.writerow(["epoch", "train_loss", "test_accuracy"])
                for ep_i, (l_val, a_val) in enumerate(zip(losses_all, accs_all), start=1):
                    cwriter.writerow([ep_i, l_val, a_val])

            if final_acc > melhor_acc:
                melhor_acc = final_acc
                melhor_config = (lr, max_epochs, hidden_size)
                melhores_params = parameters
                melhores_losses_100 = losses_100
                melhores_accs_100 = accs_100

            print(
                f"[{idx:02d}/{total}] lr={lr:<5} | epochs={max_epochs:<3} | "
                f"hidden={hidden_size:<3} -> Loss: {final_loss:.4f} | "
                f"Acc: {final_acc * 100:.2f}% ({duracao:.1f}s)"
            )

    if melhores_params is not None:
        W1, b1, W2, b2 = melhores_params
        np.save("model/W1.npy", W1)
        np.save("model/b1.npy", b1)
        np.save("model/W2.npy", W2)
        np.save("model/b2.npy", b2)
        np.savetxt("model/losses.csv", melhores_losses_100, delimiter=",")
        np.savetxt("model/accuracies.csv", melhores_accs_100, delimiter=",")

    print("\n" + "=" * 60)
    print(f"Tabela das 64 combinações salva em: {summary_path}")
    print(
        f"Melhor configuração: lr={melhor_config[0]}, epochs={melhor_config[1]}, "
        f"hidden_size={melhor_config[2]} -> Acurácia de teste: {melhor_acc * 100:.2f}%"
    )
    print("Pesos e curvas da melhor configuração salvos em model/ (prontos para plot.py).")
    print("=" * 60)

if __name__ == "__main__":
    run_grid_search()
    
    
#[47/64] lr=0.1   | epochs=500 | hidden=64  -> Loss: 0.3417 | Acc: 90.21% (123.5s)
#[48/64] lr=0.1   | epochs=500 | hidden=256 -> Loss: 0.3289 | Acc: 90.62% (489.2s)
#[56/64] lr=0.5   | epochs=100 | hidden=256 -> Loss: 0.3290 | Acc: 90.67% (91.5s)
#[58/64] lr=0.5   | epochs=200 | hidden=16  -> Loss: 0.2895 | Acc: 91.51% (31.7s)