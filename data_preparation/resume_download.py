from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import xlwt, xlrd
from xlutils.copy import copy
import random
import time
# from selenium.webdriver.common.by import By
import re


# pip --default-timeout=100 install selenium==4.1.0 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
# https://chromedriver.chromium.org/home
def main(chromedriver_path, pdb_list, save_csv):
    chrome_options = Options()
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_argument('blink-settings=imagesEnabled=false')
    # driver = webdriver.Chrome(options=chrome_options,executable_path=r'C:\Users\40763\.cache\selenium\chromedriver\win32\109.0.5414.74\chromedriver.exe')
    driver = webdriver.Chrome(options=chrome_options, executable_path=chromedriver_path)

    with open(pdb_list, 'r') as f:
        L = f.readlines()
        L = [i.rstrip().split(',') for i in L]
        pdbselected = L[0]

    data = xlrd.open_workbook(save_csv, formatting_info=True)
    work_book = copy(wb=data)
    work_sheet = work_book.get_sheet(0)
    have_save = False
    # work_book = xlwt.Workbook()
    # work_sheet = work_book.add_sheet("data")

    for i in range(len(pdbselected)):
        if i == 1155:
            have_save = False
        if i % 1000:
            print(i)
        if have_save:
            continue
        time.sleep(random.uniform(1, 3))
        url = 'http://www1.rcsb.org/structure/' + str(pdbselected[i])
        driver.get(url)

        try:
            Organism = driver.find_element_by_xpath("//*[@id='header_organism']")
            # Organism = driver.find_element(By.XPATH,"//*[@id='header_organism']")
        except Exception:
            ListOfOrganism = [' ', ' ']
        else:
            ListOfOrganism = re.split("[\n:]", Organism.text)

        try:
            Expression_System = driver.find_element_by_xpath("//*[@id='header_expression-system']")
            # Expression_System = driver.find_element(By.XPATH,"//*[@id='header_expression-system']")
        except Exception:
            ListOfExpression_System = [' ', ' ']
        else:
            ListOfExpression_System = re.split("[\n:]", Expression_System.text)

        try:
            Mutation = driver.find_element_by_xpath("//*[@id='header_mutation']")
            # Mutation = driver.find_element(By.XPATH,"//*[@id='header_mutation']")
        except Exception:
            ListOfMutation = [' ', ' ']
        else:
            ListOfMutation = re.split("[\n:]", Mutation.text)

        try:
            Released = driver.find_element_by_xpath("//*[@id='header_deposited-released-dates']")
            # Released = driver.find_element(By.XPATH,"//*[@id='header_deposited-released-dates']")
        except Exception:
            ListOfReleased = [' ', ' ', ' ']
        else:
            ListOfReleased = re.split("[\n:]", Released.text)

        try:
            Resolution = driver.find_element_by_xpath("//*[@id='exp_header_0_diffraction_resolution']")
            # Resolution = driver.find_element(By.XPATH,"//*[@id='exp_header_0_diffraction_resolution']")
        except Exception:
            ListOfResolution = [' ', ' ']
        else:
            ListOfResolution = re.split("[\n:]", Resolution.text)

        try:
            PubMed = driver.find_element_by_css_selector('#pubmedLinks a')
        except Exception:
            ListOfPubMed = [' ']
        else:
            ListOfPubMed = re.split("[\n: ]", PubMed.text)

        try:
            Molecule_1 = driver.find_element_by_css_selector('#macromolecule-entityId-1-rowDescription td:nth-child(1)')
            Chains_1 = driver.find_element_by_css_selector('#macromolecule-entityId-1-rowDescription td:nth-child(2)')
            SequenceLength_1 = driver.find_element_by_css_selector(
                '#macromolecule-entityId-1-rowDescription td:nth-child(3)')
            Organism_1 = driver.find_element_by_css_selector('#macromolecule-entityId-1-rowDescription td:nth-child(4)')
            Details_1 = driver.find_element_by_css_selector('#macromolecule-entityId-1-rowDescription td:nth-child(5)')
        except Exception:
            ListofEntity_1 = [' ', ' ', ' ', ' ', ' ']
        else:
            ListofEntity_1 = []
            ListofEntity_1.append(Molecule_1.text)
            ListofEntity_1.append(Chains_1.text)
            ListofEntity_1.append(SequenceLength_1.text)
            ListofEntity_1.append(Organism_1.text)
            ListofEntity_1.append(Details_1.text)

        try:
            UniProt_1 = driver.find_element_by_css_selector(
                '#table_macromolecule-protein-entityId-1 > tbody > tr:nth-child(5) > td >div > a')
        except Exception:
            ListOfUniProt_1 = [' ']
        else:
            ListOfUniProt_1 = re.split("[ ]", UniProt_1.text)

        try:
            Molecule_2 = driver.find_element_by_css_selector('#macromolecule-entityId-2-rowDescription td:nth-child(1)')
            Chains_2 = driver.find_element_by_css_selector('#macromolecule-entityId-2-rowDescription td:nth-child(2)')
            SequenceLength_2 = driver.find_element_by_css_selector(
                '#macromolecule-entityId-2-rowDescription td:nth-child(3)')
            Organism_2 = driver.find_element_by_css_selector('#macromolecule-entityId-2-rowDescription td:nth-child(4)')
            Details_2 = driver.find_element_by_css_selector('#macromolecule-entityId-2-rowDescription td:nth-child(5)')
        except Exception:
            ListofEntity_2 = [' ', ' ', ' ', ' ', ' ']
        else:
            ListofEntity_2 = []
            ListofEntity_2.append(Molecule_2.text)
            ListofEntity_2.append(Chains_2.text)
            ListofEntity_2.append(SequenceLength_2.text)
            ListofEntity_2.append(Organism_2.text)
            ListofEntity_2.append(Details_2.text)

        try:
            UniProt_2 = driver.find_element_by_css_selector(
                '#table_macromolecule-protein-entityId-2 > tbody > tr:nth-child(5) > td >div > a')
        except Exception:
            ListOfUniProt_2 = [' ']
        else:
            ListOfUniProt_2 = re.split("[ ]", UniProt_2.text)

        try:
            Molecule_3 = driver.find_element_by_css_selector('#macromolecule-entityId-3-rowDescription td:nth-child(1)')
            Chains_3 = driver.find_element_by_css_selector('#macromolecule-entityId-3-rowDescription td:nth-child(2)')
            SequenceLength_3 = driver.find_element_by_css_selector(
                '#macromolecule-entityId-3-rowDescription td:nth-child(3)')
            Organism_3 = driver.find_element_by_css_selector('#macromolecule-entityId-3-rowDescription td:nth-child(4)')
            Details_3 = driver.find_element_by_css_selector('#macromolecule-entityId-3-rowDescription td:nth-child(5)')
        except Exception:
            ListofEntity_3 = [' ', ' ', ' ', ' ', ' ']
        else:
            ListofEntity_3 = []
            ListofEntity_3.append(Molecule_3.text)
            ListofEntity_3.append(Chains_3.text)
            ListofEntity_3.append(SequenceLength_3.text)
            ListofEntity_3.append(Organism_3.text)
            ListofEntity_3.append(Details_3.text)

        try:
            UniProt_3 = driver.find_element_by_css_selector(
                '#table_macromolecule-protein-entityId-3 > tbody > tr:nth-child(5) > td >div > a')
        except Exception:
            ListOfUniProt_3 = [' ']
        else:
            ListOfUniProt_3 = re.split("[ ]", UniProt_3.text)

        try:
            Molecule_4 = driver.find_element_by_css_selector('#macromolecule-entityId-4-rowDescription td:nth-child(1)')
            Chains_4 = driver.find_element_by_css_selector('#macromolecule-entityId-4-rowDescription td:nth-child(2)')
            SequenceLength_4 = driver.find_element_by_css_selector(
                '#macromolecule-entityId-4-rowDescription td:nth-child(3)')
            Organism_4 = driver.find_element_by_css_selector('#macromolecule-entityId-4-rowDescription td:nth-child(4)')
            Details_4 = driver.find_element_by_css_selector('#macromolecule-entityId-4-rowDescription td:nth-child(5)')
        except Exception:
            ListofEntity_4 = [' ', ' ', ' ', ' ', ' ']
        else:
            ListofEntity_4 = []
            ListofEntity_4.append(Molecule_4.text)
            ListofEntity_4.append(Chains_4.text)
            ListofEntity_4.append(SequenceLength_4.text)
            ListofEntity_4.append(Organism_4.text)
            ListofEntity_4.append(Details_4.text)

        try:
            UniProt_4 = driver.find_element_by_css_selector(
                '#table_macromolecule-protein-entityId-4 > tbody > tr:nth-child(5) > td >div > a')
        except Exception:
            ListOfUniProt_4 = [' ']
        else:
            ListOfUniProt_4 = re.split("[ ]", UniProt_4.text)

        try:
            Molecule_5 = driver.find_element_by_css_selector('#macromolecule-entityId-5-rowDescription td:nth-child(1)')
            Chains_5 = driver.find_element_by_css_selector('#macromolecule-entityId-5-rowDescription td:nth-child(2)')
            SequenceLength_5 = driver.find_element_by_css_selector(
                '#macromolecule-entityId-5-rowDescription td:nth-child(3)')
            Organism_5 = driver.find_element_by_css_selector('#macromolecule-entityId-5-rowDescription td:nth-child(4)')
            Details_5 = driver.find_element_by_css_selector('#macromolecule-entityId-5-rowDescription td:nth-child(5)')
        except Exception:
            ListofEntity_5 = [' ', ' ', ' ', ' ', ' ']
        else:
            ListofEntity_5 = []
            ListofEntity_5.append(Molecule_5.text)
            ListofEntity_5.append(Chains_5.text)
            ListofEntity_5.append(SequenceLength_5.text)
            ListofEntity_5.append(Organism_5.text)
            ListofEntity_5.append(Details_5.text)

        try:
            UniProt_5 = driver.find_element_by_css_selector(
                '#table_macromolecule-protein-entityId-5 > tbody > tr:nth-child(5) > td >div > a')
        except Exception:
            ListOfUniProt_5 = [' ']
        else:
            ListOfUniProt_5 = re.split("[ ]", UniProt_5.text)

        try:
            Molecule_6 = driver.find_element_by_css_selector('#macromolecule-entityId-6-rowDescription td:nth-child(1)')
            Chains_6 = driver.find_element_by_css_selector('#macromolecule-entityId-6-rowDescription td:nth-child(2)')
            SequenceLength_6 = driver.find_element_by_css_selector(
                '#macromolecule-entityId-6-rowDescription td:nth-child(3)')
            Organism_6 = driver.find_element_by_css_selector('#macromolecule-entityId-6-rowDescription td:nth-child(4)')
            Details_6 = driver.find_element_by_css_selector('#macromolecule-entityId-6-rowDescription td:nth-child(5)')
        except Exception:
            ListofEntity_6 = [' ', ' ', ' ', ' ', ' ']
        else:
            ListofEntity_6 = []
            ListofEntity_6.append(Molecule_6.text)
            ListofEntity_6.append(Chains_6.text)
            ListofEntity_6.append(SequenceLength_6.text)
            ListofEntity_6.append(Organism_6.text)
            ListofEntity_6.append(Details_6.text)

        try:
            UniProt_6 = driver.find_element_by_css_selector(
                '#table_macromolecule-protein-entityId-6 > tbody > tr:nth-child(5) > td >div > a')
        except Exception:
            ListOfUniProt_6 = [' ']
        else:
            ListOfUniProt_6 = re.split("[ ]", UniProt_6.text)

        try:
            Molecule_7 = driver.find_element_by_css_selector('#macromolecule-entityId-7-rowDescription td:nth-child(1)')
            Chains_7 = driver.find_element_by_css_selector('#macromolecule-entityId-7-rowDescription td:nth-child(2)')
            SequenceLength_7 = driver.find_element_by_css_selector(
                '#macromolecule-entityId-7-rowDescription td:nth-child(3)')
            Organism_7 = driver.find_element_by_css_selector('#macromolecule-entityId-7-rowDescription td:nth-child(4)')
            Details_7 = driver.find_element_by_css_selector('#macromolecule-entityId-7-rowDescription td:nth-child(5)')
        except Exception:
            ListofEntity_7 = [' ', ' ', ' ', ' ', ' ']
        else:
            ListofEntity_7 = []
            ListofEntity_7.append(Molecule_7.text)
            ListofEntity_7.append(Chains_7.text)
            ListofEntity_7.append(SequenceLength_7.text)
            ListofEntity_7.append(Organism_7.text)
            ListofEntity_7.append(Details_7.text)

        try:
            UniProt_7 = driver.find_element_by_css_selector(
                '#table_macromolecule-protein-entityId-7 > tbody > tr:nth-child(5) > td >div > a')
        except Exception:
            ListOfUniProt_7 = [' ']
        else:
            ListOfUniProt_7 = re.split("[ ]", UniProt_7.text)

        try:
            Molecule_8 = driver.find_element_by_css_selector('#macromolecule-entityId-8-rowDescription td:nth-child(1)')
            Chains_8 = driver.find_element_by_css_selector('#macromolecule-entityId-8-rowDescription td:nth-child(2)')
            SequenceLength_8 = driver.find_element_by_css_selector(
                '#macromolecule-entityId-8-rowDescription td:nth-child(3)')
            Organism_8 = driver.find_element_by_css_selector('#macromolecule-entityId-8-rowDescription td:nth-child(4)')
            Details_8 = driver.find_element_by_css_selector('#macromolecule-entityId-8-rowDescription td:nth-child(5)')
        except Exception:
            ListofEntity_8 = [' ', ' ', ' ', ' ', ' ']
        else:
            ListofEntity_8 = []
            ListofEntity_8.append(Molecule_8.text)
            ListofEntity_8.append(Chains_8.text)
            ListofEntity_8.append(SequenceLength_8.text)
            ListofEntity_8.append(Organism_8.text)
            ListofEntity_8.append(Details_8.text)

        try:
            UniProt_8 = driver.find_element_by_css_selector(
                '#table_macromolecule-protein-entityId-8 > tbody > tr:nth-child(5) > td >div > a')
        except Exception:
            ListOfUniProt_8 = [' ']
        else:
            ListOfUniProt_8 = re.split("[ ]", UniProt_8.text)

        try:
            Molecule_9 = driver.find_element_by_css_selector('#macromolecule-entityId-9-rowDescription td:nth-child(1)')
            Chains_9 = driver.find_element_by_css_selector('#macromolecule-entityId-9-rowDescription td:nth-child(2)')
            SequenceLength_9 = driver.find_element_by_css_selector(
                '#macromolecule-entityId-9-rowDescription td:nth-child(3)')
            Organism_9 = driver.find_element_by_css_selector('#macromolecule-entityId-9-rowDescription td:nth-child(4)')
            Details_9 = driver.find_element_by_css_selector('#macromolecule-entityId-9-rowDescription td:nth-child(5)')
        except Exception:
            ListofEntity_9 = [' ', ' ', ' ', ' ', ' ']
        else:
            ListofEntity_9 = []
            ListofEntity_9.append(Molecule_9.text)
            ListofEntity_9.append(Chains_9.text)
            ListofEntity_9.append(SequenceLength_9.text)
            ListofEntity_9.append(Organism_9.text)
            ListofEntity_9.append(Details_9.text)

        try:
            UniProt_9 = driver.find_element_by_css_selector(
                '#table_macromolecule-protein-entityId-9 > tbody > tr:nth-child(5) > td >div > a')
        except Exception:
            ListOfUniProt_9 = [' ']
        else:
            ListOfUniProt_9 = re.split("[ ]", UniProt_9.text)

        ListOfContain = ListOfOrganism + ListOfExpression_System + ListOfMutation + ListOfReleased + ListOfResolution + ListOfPubMed + ListofEntity_1 + ListOfUniProt_1 + ListofEntity_2 + ListOfUniProt_2 + ListofEntity_3 + ListOfUniProt_3 + ListofEntity_4 + ListOfUniProt_4 + ListofEntity_5 + ListOfUniProt_5 + ListofEntity_6 + ListOfUniProt_6 + ListofEntity_7 + ListOfUniProt_7 + ListofEntity_8 + ListOfUniProt_8 + ListofEntity_9 + ListOfUniProt_9

        work_sheet.write(i, 0, pdbselected[i])
        for j in range(len(ListOfContain)):
            work_sheet.write(i, j + 1, ListOfContain[j])
        work_book.save('2.csv')


if __name__ == '__main__':
    main('D:\work\chromedriver.exe', 'test.txt', '2.csv')
