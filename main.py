from functions import sigmoid, compute_loss

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


if __name__ == "__main__":
    i = 0
    w1, w2 = 0.6, 0.3

    print(f"Iteration {i}")
    w1, w2, updated_loss, gradient_w1, gradient_w2 = backpropagation(x=1.7, w1=w1, w2=w2, y=2, n=0.3)
    print("---------------****-----------------****-----------------")
    i += 1

    while updated_loss > 0.6:
        print(f"Iteration {i}")
        w1, w2, updated_loss, gradient_w1, gradient_w2 = backpropagation(x=1.7, w1=w1, w2=w2, y=2, n=0.3)
        if gradient_w1 == 0.0 or gradient_w2 == 0.0 or gradient_w1 == -0.0 or gradient_w2 == -0.0:
            print("Gradient is zero, stopping the training.")
            break
        print("---------------****-----------------****-----------------")
        i += 1

    print(f"Final Weights: w1 = {w1:.2f}, w2 = {w2:.2f}")
