if __name__ == '__main__':
    file1 = open("/h/u15/c4/00/sunchuan/01")
    sth = file1.readline()
    while sth:
        print(sth)
        sth = file1.readline()
    print(sth)
