import xlrd  # 导入xlrd模块
import xlwt  # 导入xlwt模块
import re
def is2022(date):
    if date[0] == ' 2023':
        return '1'
    elif date[0] == ' 2022' and date[1] == '12':
        return '1'
    else:
        return '0'
def is_antibody(lis):
    '''
    Determine whether it is antibody    Lysine/arginine/ornithine transport protein
    '''
    num = 0
    if lis == ' ':
        return '2'
    pattern1 = re.compile('antibody+', 2)
    pattern2 = re.compile('Fv+', 2)
    pattern3 = re.compile('Heavy chain  +', 2)
    pattern4 = re.compile('Light chain+', 2)
    pattern5 = re.compile('Fab+', 2)
    pattern6 = re.compile('VHH+', 2)
    pattern7 = re.compile('IMMUNOGLOBULIN+', 2)
    pattern8 = re.compile('IG +', 2)
    pattern9 = re.compile(' IG+', 2)
    pattern10 = re.compile('FAB+', 2)
    pattern11 = re.compile('FV+')
    if pattern1.search(lis) or pattern2.search(lis) or pattern3.search(lis) or pattern4.search(lis) \
            or pattern5.search(lis) or pattern6.search(lis) or pattern7.search(lis) or pattern8.search(lis) \
            or pattern9.search(lis) or pattern10.search(lis) or pattern11.search(lis):
        return '0'
    return '1'

def select_data1(num):
    # 1. 打开表格
    excel_ = xlrd.open_workbook('./'+str(num)+'.csv')  # 打开表格

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
    # 1.新建工作簿
    workbook = xlwt.Workbook()

    # i=1
    # 2.新建工作表并重命名
    worksheet = workbook.add_sheet('data')
    # 3.写入内容
    worksheet.write(0, 0, 'ID')
    for j in range(1, 4):
        worksheet.write(0, j, Table.cell_value(0, j * 2 - 1))
    worksheet.write(0, 4, 'Released')
    worksheet.write(0, 5, 'Resolution')
    worksheet.write(0, 6, 'PubMed')
    for i in range(1, 6):
        worksheet.write(0, i * 7 + 0, 'Molecule' + str(i))
        worksheet.write(0, i * 7 + 1, 'flag')
        worksheet.write(0, i * 7 + 2, 'Chains')
        worksheet.write(0, i * 7 + 3, 'Sequence Length')
        worksheet.write(0, i * 7 + 4, 'Details')
        worksheet.write(0, i * 7 + 5, 'Organism')
        worksheet.write(0, i * 7 + 6, 'UniProt')

    for i in range(1, nrows):
        worksheet.write(i, 0, Table.cell_value(i, 0))
        for j in range(1, 4):
            worksheet.write(i, j, Table.cell_value(i, j * 2))
        worksheet.write(i, 4, Table.cell_value(i, 9))
        worksheet.write(i, 5, Table.cell_value(i, 11))
        worksheet.write(i, 6, Table.cell_value(i, 12))
        for molecole in range(1, 6):
            if is_antibody(Table.cell_value(i, (molecole-1) * 6 + 13)) == '1':
                worksheet.write(i, molecole * 7 + 0, Table.cell_value(i, (molecole - 1) * 6 + 13))
                worksheet.write(i, molecole * 7 + 1, is_antibody(Table.cell_value(i, (molecole-1) * 6 + 13)))
                worksheet.write(i, molecole * 7 + 2, Table.cell_value(i, (molecole-1) * 6 + 14))
                worksheet.write(i, molecole * 7 + 3, Table.cell_value(i, (molecole-1) * 6 + 15))
                worksheet.write(i, molecole * 7 + 4, Table.cell_value(i, (molecole-1) * 6 + 16))
                worksheet.write(i, molecole * 7 + 5, Table.cell_value(i, (molecole-1) * 6 + 17))
                worksheet.write(i, molecole * 7 + 6, Table.cell_value(i, (molecole-1) * 6 + 18))
            elif is_antibody(Table.cell_value(i, (molecole-1) * 6 + 13)) == '0':
                worksheet.write(i, molecole * 7 + 0, Table.cell_value(i, (molecole - 1) * 6 + 13))
                worksheet.write(i, molecole * 7 + 1, is_antibody(Table.cell_value(i, (molecole-1) * 6 + 13)))
                worksheet.write(i, molecole * 7 + 2, Table.cell_value(i, (molecole-1) * 6 + 14))
                worksheet.write(i, molecole * 7 + 3, Table.cell_value(i, (molecole-1) * 6 + 15))
                worksheet.write(i, molecole * 7 + 4, Table.cell_value(i, (molecole-1) * 6 + 16))
                worksheet.write(i, molecole * 7 + 5, Table.cell_value(i, (molecole-1) * 6 + 17))
                worksheet.write(i, molecole * 7 + 6, Table.cell_value(i, (molecole-1) * 6 + 18))
    workbook.save('./data'+str(num)+'.xls')

def select_data2(input,output,after2023,len50):
    # 1. 打开表格
    excel_ = xlrd.open_workbook(input)  # 打开表格
    # 2.定位
    Table = excel_.sheet_by_index(0)  # 通过索引定位工作表，索引从0开始
    Table_1 = excel_.sheet_by_name('data')  # 通过表的名字定位工作表
    # 3. 读取单元格
    ncols = Table.ncols  # 列
    nrows = Table.nrows  # 行

    # 1.新建工作簿
    workbook = xlwt.Workbook()
    # 2.新建工作表并重命名
    worksheet = workbook.add_sheet('data')
    # 3.写入内容
    worksheet.write(0, 0, 'ID')
    for j in range(1, 4):
        worksheet.write(0, j, Table.cell_value(0, j * 2 - 1))
    worksheet.write(0, 4, 'Released')
    worksheet.write(0, 5, 'Resolution')
    worksheet.write(0, 6, 'PubMed')
    for i in range(1, 6):
        worksheet.write(0, i * 7 + 0, 'Molecule' + str(i))
        worksheet.write(0, i * 7 + 1, 'flag')
        worksheet.write(0, i * 7 + 2, 'Chains')
        worksheet.write(0, i * 7 + 3, 'Sequence Length')
        worksheet.write(0, i * 7 + 4, 'Details')
        worksheet.write(0, i * 7 + 5, 'Organism')
        worksheet.write(0, i * 7 + 6, 'UniProt')
    count = 1
    if after2023 == 1 and len50 == 1:
        for i in range(1, nrows):
            flag1 = flag2 = 0
            len = Table.cell_value(i, 10)
            date = Table.cell_value(i, 4)
            date = date.split('-')
            try:
                # after2023 1 before2023 0
                if int(len) > 50 and is2022(date) == '1':
                    for molecole in range(1, 6):
                        if Table.cell_value(i, molecole * 7 + 1) == '1':
                            flag1 = 1
                        elif Table.cell_value(i, molecole * 7 + 1) == '0':
                            flag2 = 1
                    if flag1 == 1 and flag2 == 1:
                        for j in range(0, ncols):
                            worksheet.write(count, j, Table.cell_value(i, j))
                        count += 1
            except:
                continue

        workbook.save(output)
    elif after2023 == 1 and len50 == 0:
        for i in range(1, nrows):
            flag1 = flag2 = 0
            len = Table.cell_value(i, 10)
            date = Table.cell_value(i, 4)
            date = date.split('-')
            try:
                # after2023 1 before2023 0
                if int(len) <= 50 and is2022(date) == '1':
                    for molecole in range(1, 6):
                        if Table.cell_value(i, molecole * 7 + 1) == '1':
                            flag1 = 1
                        elif Table.cell_value(i, molecole * 7 + 1) == '0':
                            flag2 = 1
                    if flag1 == 1 and flag2 == 1:
                        for j in range(0, ncols):
                            worksheet.write(count, j, Table.cell_value(i, j))
                        count += 1
            except:
                continue
        workbook.save(output)

    elif after2023 == 0 and len50 == 1:
        for i in range(1, nrows):
            flag1 = flag2 = 0
            len = Table.cell_value(i, 10)
            date = Table.cell_value(i, 4)
            date = date.split('-')
            try:
                # after2023 1 before2023 0
                if int(len) > 50 and is2022(date) == '0':
                    for molecole in range(1, 6):
                        if Table.cell_value(i, molecole * 7 + 1) == '1':
                            flag1 = 1
                        elif Table.cell_value(i, molecole * 7 + 1) == '0':
                            flag2 = 1
                    if flag1 == 1 and flag2 == 1:
                        for j in range(0, ncols):
                            worksheet.write(count, j, Table.cell_value(i, j))
                        count += 1
            except:
                continue
        workbook.save(output)
    elif after2023 == 0 and len50 == 0:
        for i in range(1, nrows):
            flag1 = flag2 = 0
            len = Table.cell_value(i, 10)
            date = Table.cell_value(i, 4)
            date = date.split('-')
            try:
                # after2023 1 before2023 0
                if int(len) <= 50 and is2022(date) == '0':
                    for molecole in range(1, 6):
                        if Table.cell_value(i, molecole * 7 + 1) == '1':
                            flag1 = 1
                        elif Table.cell_value(i, molecole * 7 + 1) == '0':
                            flag2 = 1
                    if flag1 == 1 and flag2 == 1:
                        for j in range(0, ncols):
                            worksheet.write(count, j, Table.cell_value(i, j))
                        count += 1
            except:
                continue
        workbook.save(output)

#select_data1(5)
select_data2('data2.xls','./lenhigher50date2023before.xls',len50=1,after2023=0)
# 1
# 2
# 3
# 4