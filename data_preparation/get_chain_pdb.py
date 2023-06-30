import xlrd
import os


def get_true_chain(chain):
    if chain == '':
        return None
    if len(chain) > 1:
        return chain[-2].upper()
    return chain.upper()


def get_chain_list(line):
    chain_list = []
    chain_site = 9
    for i in range(5):
        if line[chain_site] != '':
            for chain in line[chain_site].split(','):
                if chain.strip() == '':
                    continue
                chain_list.extend(get_true_chain(chain.strip()))
        chain_site += 7
    return chain_list


def get_pdb_by_chain(name, chain, xls_name):
    with open('pdb/' + xls_name + '/' + name + '.pdb') as r:
        with open('pdb_chain/' + xls_name + '/' + name + '_' + chain + '.pdb', 'w') as f:
            for line in r:
                row = line.strip().split()
                if (row[0] == 'ATOM' and row[4][0] == chain):
                    f.write(line)


def main():
    xls_name = 'lenhigher50date2023before_less3A'
    data = xlrd.open_workbook('file_recv/' + xls_name + '.xlsx')
    table = data.sheets()[0]
    table.col_values(0, start_rowx=0, end_rowx=None)[1:]
    nrows = table.nrows
    if not os.path.exists('pdb_chain/' + xls_name):
        os.mkdir('pdb_chain/' + xls_name)
    for i in range(1, nrows):
        lines = table.row_values(i, start_colx=0, end_colx=None)
        chain_list = get_chain_list(lines)
        for chain in chain_list:
            get_pdb_by_chain(lines[0], chain, xls_name)
        print(lines[0])
        print(chain_list)


if __name__ == '__main__':
    main()
