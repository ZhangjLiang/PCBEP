import requests
import xlrd
import os
from bs4 import BeautifulSoup


def download_content(url):
    """
    第一个函数，用来下载网页，返回网页内容
    参数 url 代表所要下载的网页网址。
    整体代码和之前类似
    """
    response = requests.get(url).text
    return response


# 第二个函数，将字符串内容保存到文件中
# 第一个参数为所要保存的文件名，第二个参数为要保存的字符串内容的变量
def save_to_file(filename, content, file_path):
    if content.startswith('<!DOCTYPE HTML PUBLIC') or content.startswith('<html>'):
        return None
    with open(file_path + '/' + filename, mode="w", encoding="utf-8") as f:
        f.write(content)
    return filename


def create_doc_from_filename(filename):
    # 输入参数为要分析的 html 文件名，返回值为对应的 BeautifulSoup 对象
    with open(filename, "r", encoding='utf-8') as f:
        html_content = f.read()
        soup = BeautifulSoup(html_content, "lxml")
    return soup


def parse(soup):
    post_list = soup.find_all("div", class_="post-info")
    for post in post_list:
        link = post.find_all("a")[1]
        print(link.text.strip())
        print(link["href"])


def get_pdb_name_list(list_path):
    pdb_list = []
    with open(list_path) as fw:
        for line in fw:
            pdb_list.append(line.strip().split()[0])
    return pdb_list


def main(list_path, file_path):
    pdb_name_list = get_pdb_name_list(list_path)
    print('download numbers: {}'.format(len(pdb_name_list)))
    print('start download')
    num = 0
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    have_pdb_list = []
    for name in pdb_name_list:
        if num % 20 == 0:
            print('has been down: {}'.format(num))
        url = "https://files.rcsb.org/view/" + name + ".pdb"
        filename = name + ".pdb"
        result = download_content(url)
        name_flag = save_to_file(filename, result, file_path)
        if name_flag:
            have_pdb_list.append(name_flag[:-4])
        num += 1
    print("finish down")

    print('update file: {}'.format(list_path))
    all_write = ''
    with open(list_path) as fw:
        for line in fw:
            if line.strip().split()[0] in have_pdb_list:
                all_write += line
    with open(list_path, 'w') as fw:
        fw.write(all_write)
    print('done update file: {}'.format(list_path))
    # soup = create_doc_from_filename(filename)
    # parse(soup)


if __name__ == '__main__':
    data = xlrd.open_workbook('file_recv/lenhigher50date2023before_less3A.xlsx')
    table = data.sheets()[0]
    pdb_name_list = table.col_values(0, start_rowx=0, end_rowx=None)[1:]
    main(pdb_name_list, 'lenhigher50date2023before_less3A')
