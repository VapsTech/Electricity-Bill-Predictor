import matplotlib.pyplot as plt

def graph(data, predictions):
    plt.scatter(data['Usage'], data['Charge'], color='red', label="Actual Values")
    plt.scatter(data['Usage'], predictions, color='blue', label="Predicted Values")

    plt.title("Electricity Bill Predictor")
    plt.xlabel("Power Consumption(kWh)")
    plt.ylabel("Price(USD)")

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.legend()
    plt.show()
