from functions import sigmoid, compute_loss, derivation_sigmoid

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
    i = 1
    x = 0.7
    w11, w12, w21, w22 = 0.5, 0.6, 0.7, 0.8
    y = 1.0
    learning_rate = 0.1
    w11, w12, w21, w22, loss, w11_gradient, w12_gradient, w21_gradient, w22_gradient = backpropagation_with_2_hl(x=x, w11=w11, w12=w12, w21=w21, w22=w22, y=y, learning_rate=learning_rate)
    while loss > 0.005:
        print(f"Iteration {i}")
        w11, w12, w21, w22, loss, w11_gradient, w12_gradient, w21_gradient, w22_gradient = backpropagation_with_2_hl(x=x, w11=w11, w12=w12, w21=w21, w22=w22, y=y, learning_rate=learning_rate)
        if w11_gradient == 0.0 or w12_gradient == 0.0 or w21_gradient == 0.0 or w22_gradient == 0.0 or w11_gradient == -0.0 or w12_gradient == -0.0 or w21_gradient == -0.0 or w22_gradient == -0.0:
            print("Gradient is zero, stopping the training.")
            break
        print("---------------****-----------------****-----------------")
        i += 1
    print(f"Training completed after {i} iterations.")
    print(f"Final Weights: w1 = {w11:.2f}, w2 = {w12:.2f}, w21 = {w21:.2f}, w22 = {w22:.2f}")
    print(f"Final Loss: {loss:.3f}")
    i += 1