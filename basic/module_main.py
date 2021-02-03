
def main():
    print("First Module's name: {}".format(__name__))


if __name__ == '__main__':
    main()
    print('Run Directly')
else:
    print('Run import')
