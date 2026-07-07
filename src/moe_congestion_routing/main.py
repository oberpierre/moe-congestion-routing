def greet(name: str = "moe-congestion-routing") -> str:
    return f"Hello from {name}!"


def main() -> None:
    print(greet())


if __name__ == "__main__":
    main()
