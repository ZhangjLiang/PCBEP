import xlrd
data = xlrd.open_workbook('file_recv/lenhigher50date2023.xls')
table = data.sheets()[0]
table.col_values(0, start_rowx=0, end_rowx=None)[1:]
# print(table.col_values(0, start_rowx=0, end_rowx=None)[1:])