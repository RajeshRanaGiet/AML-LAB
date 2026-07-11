from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import numpy as np
import os
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/visualize_regression", methods=["POST"])
def visualize_regression():
    data = request.get_json()
    x = np.array(data["x"])
    y = np.array(data["y"])
    m = data["m"]
    b = data["b"]
    y_pred = []
    for i in range(len(x)):
        y_pred.append(m * x[i] + b)
    plt.scatter(x, y, label="Data Points")
    plt.plot(x, y_pred, color="red", label="Regression Line")
    plt.title("Linear Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    image1_path = "static/regression_plot.png"
    plt.savefig(image1_path)
    plt.close()

    n = len(x)
    mae = 0
    mse = 0
    for i in range(n):
        error = y[i] - y_pred[i]
        mae += abs(error)
        mse += error ** 2
    mae = mae / n
    mse = mse / n
    rmse = mse ** 0.5
    mean_y = sum(y) / n
    ss_res = 0
    ss_tot = 0
    for i in range(n):
        ss_res += (y[i] - y_pred[i]) ** 2
        ss_tot += (y[i] - mean_y) ** 2
    r2 = 1 - (ss_res / ss_tot)

    metrics = ["MSE", "MAE", "RMSE", "R2"]
    values = [mse, mae, rmse, r2]
    plt.figure(figsize=(6,4))
    plt.scatter(metrics, values)
    for i in range(4):
        plt.text(metrics[i], values[i], round(values[i], 4))
    plt.title("Model Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.grid(True)
    image2_path = "static/metrics_plot.png"
    plt.savefig(image2_path)
    plt.close()

    return jsonify({
        "image1": image1_path,
        "image2": image2_path
    })

if __name__ == "__main__":
    app.run(debug=True)