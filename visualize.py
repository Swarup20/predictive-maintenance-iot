import os
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/housing.csv"
OUTPUT_DIR = "outputs/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)


def plot_price_distribution():
    plt.figure(figsize=(10, 6))
    plt.hist(df["price"], bins=50)

    plt.title("Property Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Number of Properties")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "price_distribution.png")
    )
    plt.close()


def plot_price_vs_area():
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["sqft_living"],
        df["price"],
        alpha=0.5
    )

    plt.title("Property Price vs Living Area")
    plt.xlabel("Living Area (sqft)")
    plt.ylabel("Price")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "price_vs_area.png")
    )
    plt.close()


def plot_geographical_distribution():
    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["price"],
        alpha=0.6
    )

    plt.colorbar(scatter, label="Property Price")

    plt.title("Geographical Distribution of Property Prices")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "geographical_price_distribution.png")
    )
    plt.close()


def plot_price_by_bedrooms():
    bedroom_prices = df.groupby("bedrooms")["price"].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(
        bedroom_prices.index,
        bedroom_prices.values
    )

    plt.title("Average Property Price by Number of Bedrooms")
    plt.xlabel("Bedrooms")
    plt.ylabel("Average Price")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "price_by_bedrooms.png")
    )
    plt.close()


def generate_visualizations():
    print("Generating visualizations...")

    plot_price_distribution()
    plot_price_vs_area()
    plot_geographical_distribution()
    plot_price_by_bedrooms()

    print("Visualizations generated successfully.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_visualizations()
