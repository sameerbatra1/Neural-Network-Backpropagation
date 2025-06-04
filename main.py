from ActivationFunctions import sigmoid, derivation_sigmoid
from Network import backpropagation, backpropagation_with_2_hl, n_layers, plot_gradient_histories, plot_loss_history

if __name__ == "__main__":
    # i = 0
    # w1, w2 = 0.6, 0.3

    # print(f"Iteration {i}")
    # w1, w2, updated_loss, gradient_w1, gradient_w2 = backpropagation(x=1.7, w1=w1, w2=w2, y=2, n=0.3)
    # print("---------------****-----------------****-----------------")
    # i += 1

    # while updated_loss > 0.6:
    #     print(f"Iteration {i}")
    #     w1, w2, updated_loss, gradient_w1, gradient_w2 = backpropagation(x=1.7, w1=w1, w2=w2, y=2, n=0.3)
    #     if gradient_w1 == 0.0 or gradient_w2 == 0.0 or gradient_w1 == -0.0 or gradient_w2 == -0.0:
    #         print("Gradient is zero, stopping the training.")
    #         break
    #     print("---------------****-----------------****-----------------")
    #     i += 1

    # print(f"Final Weights: w1 = {w1:.2f}, w2 = {w2:.2f}")

    # ------------------------------ Hidden Layer with 2 neurons implementation ------------------------------
    # i = 1
    # x = 0.7
    # w11, w12, w21, w22 = 0.5, 0.6, 0.7, 0.8
    # y = 1.0
    # learning_rate = 0.1
    # w11, w12, w21, w22, loss, w11_gradient, w12_gradient, w21_gradient, w22_gradient = backpropagation_with_2_hl(x=x, w11=w11, w12=w12, w21=w21, w22=w22, y=y, learning_rate=learning_rate)
    # while loss > 0.005:
    #     print(f"Iteration {i}")
    #     w11, w12, w21, w22, loss, w11_gradient, w12_gradient, w21_gradient, w22_gradient = backpropagation_with_2_hl(x=x, w11=w11, w12=w12, w21=w21, w22=w22, y=y, learning_rate=learning_rate)
    #     if w11_gradient == 0.0 or w12_gradient == 0.0 or w21_gradient == 0.0 or w22_gradient == 0.0 or w11_gradient == -0.0 or w12_gradient == -0.0 or w21_gradient == -0.0 or w22_gradient == -0.0:
    #         print("Gradient is zero, stopping the training.")
    #         break
    #     print("---------------****-----------------****-----------------")
    #     i += 1
    # print(f"Training completed after {i} iterations.")
    # print(f"Final Weights: w1 = {w11:.2f}, w2 = {w12:.2f}, w21 = {w21:.2f}, w22 = {w22:.2f}")
    # print(f"Final Loss: {loss:.3f}")
    # i += 1

    # ------------------------------ N Layers implementation ------------------------------
    i = 0
    j = 0
    x = 0.7
    w_hidden = [0.1,0.2,0.3,0.8,0.9]
    b_hidden = [0.0, 0.0, 0.0, 0.0, 0.0]
    w_output = [0.2,0.5,0.1,0.6,0.7]
    b_output = 0.0
    y = 1.0
    learning_rate = 0.1
    gradient_w_out_history = [[] for _ in range(len(w_hidden))]
    gradient_w_in_history = [[] for _ in range(len(w_hidden))]
    gradient_b_hidden_history = [[] for _ in range(len(w_hidden))]
    gradient_b_output_history = []
    loss_history = []
    w_hidden, b_hidden, w_output, b_output, loss, gradient_w_in, gradient_w_out, gradient_b_hidden, gradient_b_output = n_layers(
        x=x,
        w_hidden=w_hidden,
        b_hidden=b_hidden,
        w_output=w_output,
        b_output=b_output,
        y=y,
        learning_rate=learning_rate
    )
    loss_history.append(loss)
    while loss > 0.005:
        print(f"Iteration {i}")
        w_hidden, b_hidden, w_output, b_output, loss, gradient_w_in, gradient_w_out, gradient_b_hidden, gradient_b_output = n_layers(
            x=x,
            w_hidden=w_hidden,
            b_hidden=b_hidden,
            w_output=w_output,
            b_output=b_output,
            y=y,
            learning_rate=learning_rate
        )
        loss_history.append(loss)
        i += 1
        for j in range(len(w_hidden)):
            gradient_w_out_history[j].append(gradient_w_out[j])
            gradient_w_in_history[j].append(gradient_w_in[j])
            gradient_b_hidden_history[j].append(gradient_b_hidden[j])
        gradient_b_output_history.append(gradient_b_output)
        if any(g == 0.0 or g == -0.0 for g in gradient_w_in + gradient_w_out):
            print("Gradient is zero, stopping the training.")
            break
        print("---------------****-----------------****-----------------")

    # print(f"Gradient gradient_w_out_history: {gradient_w_out_history}")
    
    plot_gradient_histories(
        gradient_w_out_history,
        gradient_w_in_history,
        gradient_b_hidden_history,
        gradient_b_output_history
    )
    plot_loss_history(loss_history)
