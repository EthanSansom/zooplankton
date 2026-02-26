from pathlib import Path


def main():
    dir = Path(__file__).parent.parent.resolve()
    print("Hello from Python")
    print(f"Project directory is: {dir}")


if __name__ == "__main__":
    main()
