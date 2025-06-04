from ActivationFunctions import sigmoid, derivation_sigmoid
import matplotlib.pyplot as plt

def compute_loss(y_true, y_pred):
    loss = 0.5 * (y_true - y_pred) ** 2
    return round(loss, 3)

def backpropagation(x, w1, w2, y, n):
    print("Step 1: Forward Pass")
    print("1.1: Hidden Layer Activation")
    z1 = round(x * w1, 3)
    print(f"z1 = {z1}")
    a1 = round(sigmoid(z1), 3)
    print(f"Hidden Layer Activation = {a1}")

    print("1.2: Output Layer Activation")
    z2 = round(a1 * w2, 3)
    a2 = round(sigmoid(z2), 3)
    print(f"z2 = {z2}")
    print(f"Output Layer Activation = {a2}")

    print("Step 2: Compute Loss")
    loss = compute_loss(y, a2)
    print(f"Loss = {loss}")

    print("Step 3: Backward Pass")
    error_at_output = round(a2 - y, 3)
    derivative_of_sigmoid = round(a2 * (1 - a2), 3)

    print("Gradient of w2")
    gradient_w2 = round(error_at_output * derivative_of_sigmoid * a1, 3)
    print(f"Gradient w2 = {gradient_w2}")

    print("Gradient of w1")
    derivative_of_sigmoid_a1 = round(a1 * (1 - a1), 3)
    gradient_w1 = round(error_at_output * derivative_of_sigmoid * w2 * derivative_of_sigmoid_a1 * x, 2)
    print(f"Gradient w1 = {gradient_w1}")

    print("Step 4: Update Weights")
    w2 = round(w2 - n * gradient_w2, 3)
    w1 = round(w1 - n * gradient_w1, 3)
    print(f"Updated w2 = {w2}")
    print(f"Updated w1 = {w1}")

    print("Step 5: Calculating updated loss")
    z1 = round(x * w1, 3)
    a1 = round(sigmoid(z1), 3)
    z2 = round(a1 * w2, 3)
    a2 = round(sigmoid(z2), 3)
    updated_loss = compute_loss(y, a2)
    print(f"Updated Loss = {updated_loss}")

    return w1, w2, updated_loss, gradient_w1, gradient_w2


def backpropagation_with_2_hl(x, w11, w12, w21, w22, y, learning_rate):
    print("Step 1: Forward Pass")
    z11 = x * w11
    a11 = sigmoid(z11)
    print(f"z11 = {round(z11, 3)}, a11 = {round(a11, 3)}")

    z12 = x * w12
    a12 = sigmoid(z12)
    print(f"z12 = {round(z12, 3)}, a12 = {round(a12, 3)}")

    z21 = a11 * w21
    z22 = a12 * w22
    zo = z21 + z22
    ao = sigmoid(zo)
    print(f"z21 = {round(z21, 3)}, z22 = {round(z22, 3)}, zo = {round(zo, 3)}, ao = {round(ao, 3)}")

    loss = compute_loss(y, ao)
    print(f"Step 2: Loss = {round(loss, 4)}")

    print("Step 3: Backward Pass")
    error_at_output = ao - y
    d_sigmoid_output = derivation_sigmoid(ao)
    s3 = error_at_output * d_sigmoid_output

    w21_gradient = s3 * a11
    w22_gradient = s3 * a12
    print(f"Gradient w21 = {round(w21_gradient, 4)}, Gradient w22 = {round(w22_gradient, 4)}")

    s1 = s3 * w21 * derivation_sigmoid(a11)
    s2 = s3 * w22 * derivation_sigmoid(a12)

    w11_gradient = s1 * x
    w12_gradient = s2 * x
    print(f"Gradient w11 = {round(w11_gradient, 4)}, Gradient w12 = {round(w12_gradient, 4)}")

    print("Step 4: Update Weights")
    w11 -= learning_rate * w11_gradient
    w12 -= learning_rate * w12_gradient
    w21 -= learning_rate * w21_gradient
    w22 -= learning_rate * w22_gradient

    print(f"Updated w11 = {round(w11, 3)}, w12 = {round(w12, 3)}, w21 = {round(w21, 3)}, w22 = {round(w22, 3)}")

    print("Step 5: Updated Forward Pass and Loss")
    z11 = x * w11
    a11 = sigmoid(z11)
    z12 = x * w12
    a12 = sigmoid(z12)
    zo = a11 * w21 + a12 * w22
    ao = sigmoid(zo)
    updated_loss = compute_loss(y, ao)
    print(f"Updated Loss = {round(updated_loss, 4)}")

    return w11, w12, w21, w22, updated_loss, w11_gradient, w12_gradient, w21_gradient, w22_gradient


def n_layers(x, w_hidden: list, b_hidden: list, w_output: list, b_output: float, y, learning_rate):
    """
    Implements a simple feedforward neural network with one hidden layer and n neurons.
    Args:
        x (float): Input value.
        w_hidden (list): Weights for the hidden layer neurons.
        b_hidden (list): Biases for the hidden layer neurons.
        w_output (list): Weights for the output layer neurons.
        b_output (float): Bias for the output layer neuron.
        y (float): Target output value.
        learning_rate (float): Learning rate for weight updates.
    Returns:
        tuple: Updated weights and biases, loss, and gradients.
    Raises:
        ValueError: If the lengths of w_hidden, w_output, and b_hidden do not match.
    """

    if len(w_hidden) != len(w_output) or len(w_hidden) != len(b_hidden):
        raise ValueError("Length mismatch: w_hidden, w_output, and b_hidden must be of the same length.")

    n = len(w_hidden)
    z = []
    a = []
    forward_pass_output = []
    gradient_w_out = []
    s_hidden = []
    gradient_w_in = []
    gradient_b_hidden = []
    gradient_b_output = 0.0

    # Step 1: Forward Pass
    print("Step 1: Forward Pass")
    print("1.1: Hidden Layer Activation")
    for i in range(n):
        z_i = x * w_hidden[i] + b_hidden[i]         # include bias
        a_i = sigmoid(z_i)
        z.append(z_i)
        a.append(a_i)
        print(f"z{i+1} = {z_i:.6f}, a{i+1} = {a_i:.6f}")

    print("1.2: Output Layer Activation")
    for i in range(n):
        forward_pass_output.append(a[i] * w_output[i])
    zo = sum(forward_pass_output) + b_output         # include output bias
    ao = sigmoid(zo)
    print(f"Output Layer Activation: zo = {zo:.6f}, ao = {ao:.6f}")

    # Step 2: Compute Loss
    print("Step 2: Compute Loss")
    loss = compute_loss(y, ao)
    print(f"Loss = {loss:.6f}")

    # Step 3: Backward Pass
    print("Step 3: Backward Pass")
    error_at_output = ao - y
    d_sigmoid_output = derivation_sigmoid(ao)
    s3 = error_at_output * d_sigmoid_output
    gradient_b_output = s3  # derivative of bias in output neuron

    for i in range(n):
        grad_out = s3 * a[i]
        gradient_w_out.append(grad_out)

        s_i = s3 * w_output[i] * derivation_sigmoid(a[i])
        s_hidden.append(s_i)

        grad_in = s_i * x
        gradient_w_in.append(grad_in)

        grad_bias_hidden = s_i
        gradient_b_hidden.append(grad_bias_hidden)

    print("\n--- Gradients ---")
    for i in range(n):
        print(f"  ∂L/∂w_hidden[{i}] = {gradient_w_in[i]:.6f}   "
              f"∂L/∂b_hidden[{i}] = {gradient_b_hidden[i]:.6f}   "
              f"∂L/∂w_output[{i}] = {gradient_w_out[i]:.6f}")
    print(f"  ∂L/∂b_output = {gradient_b_output:.6f}")

    # Step 4: Update Weights and Biases
    print("Step 4: Update Weights and Biases")
    for i in range(n):
        w_hidden[i] -= learning_rate * gradient_w_in[i]
        b_hidden[i] -= learning_rate * gradient_b_hidden[i]
        w_output[i] -= learning_rate * gradient_w_out[i]
    b_output -= learning_rate * gradient_b_output

    print("\n--- Updated weights and biases ---")
    for i in range(n):
        print(f"  w_hidden[{i}] = {w_hidden[i]:.6f}   "
              f"b_hidden[{i}] = {b_hidden[i]:.6f}   "
              f"w_output[{i}] = {w_output[i]:.6f}")
    print(f"  b_output = {b_output:.6f}")

    # Step 5: Updated Forward Pass and Loss
    print("Step 5: Updated Forward Pass and Loss")
    z = []
    a = []
    forward_pass_output = []

    for i in range(n):
        z_i = x * w_hidden[i] + b_hidden[i]
        a_i = sigmoid(z_i)
        z.append(z_i)
        a.append(a_i)
        forward_pass_output.append(a_i * w_output[i])

    zo = sum(forward_pass_output) + b_output
    ao = sigmoid(zo)
    loss = compute_loss(y, ao)
    print(f"Updated Loss = {loss:.6f}")

    return w_hidden, b_hidden, w_output, b_output, loss, gradient_w_in, gradient_w_out, gradient_b_hidden, gradient_b_output

def plot_gradient_histories(gradient_w_out_history, gradient_w_in_history,
                             gradient_b_hidden_history, gradient_b_output_history):
    plt.figure(figsize=(14, 8))

    # Plot weight gradients (output layer)
    for i, grad_list in enumerate(gradient_w_out_history):
        plt.plot(grad_list, label=f'w_output[{i}]')

    # Plot weight gradients (input to hidden)
    for i, grad_list in enumerate(gradient_w_in_history):
        plt.plot(grad_list, linestyle='--', label=f'w_hidden[{i}]')

    # Plot bias gradients (hidden layer)
    for i, grad_list in enumerate(gradient_b_hidden_history):
        plt.plot(grad_list, linestyle='-.', label=f'b_hidden[{i}]')

    # Plot bias gradients (output layer)
    plt.plot(gradient_b_output_history, linestyle=':', color='black', linewidth=2, label='b_output')

    plt.title("Gradient Histories Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Value")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, marker='o', color='blue', label='Loss')
    plt.title("Loss Over Training Iterations")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()