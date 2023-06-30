import xlrd  # 导入xlrd模块
import xlwt  # 导入xlwt模块
import re


def get_true_chain(chain):
    if chain == '':
        return None
    if len(chain) > 1:
        return chain[-2]
    return chain.upper()


def get_data(data_path, save_path):
    '''
    init list
    '''
    excel_ = xlrd.open_workbook(data_path)  # 打开表格

    # 2.定位
    Table = excel_.sheet_by_index(0)  # 通过索引定位工作表，索引从0开始
    Table_1 = excel_.sheet_by_name('data')  # 通过表的名字定位工作表

    # 3. 读取单元格
    # if len(Table.cell_value(1,8)) == 0:
    #     print('true')
    # else:
    #     print('flase')
    ncols = Table.ncols  # 列
    nrows = Table.nrows  # 行
    print('column number: {}  row number: {}'.format(ncols, nrows))
    flag_site = [8, 15, 22, 29, 36]
    all_list = ''
    for i in range(1, nrows):
        chain = ''
        flag_num = []
        for site in range(len(flag_site)):
            flag = Table.cell_value(i, flag_site[site])
            if flag in ['1', '0']:
                flag_num.append(flag)
                if flag == '1':
                    chain = Table.cell_value(i, flag_site[site] + 1)
        if '0' in flag_num and chain != '':
            # print(Table.cell_value(i, 0), get_true_chain(chain.split(',')[0]))
            all_list = all_list + Table.cell_value(i, 0) + ' ' + get_true_chain(chain.split(',')[0]) + '\n'
    # print(all_list)
    with open(save_path, 'w') as li:
        li.write(all_list)


if __name__ == '__main__':
    get_data('data.xls', 'list.txt')
