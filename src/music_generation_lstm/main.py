import cli

# expected entry point if executed as an installed package
def main():
    cli.start_session()

# entry point for simple python script execution
if __name__ == "__main__":
    main()