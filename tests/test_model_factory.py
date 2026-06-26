from src.models.model_factory import get_models


def main():

    models = get_models()

    print("Available Models:\n")

    for name in models:

        print(name)


if __name__ == "__main__":

    main()