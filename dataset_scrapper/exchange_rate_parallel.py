from selenium import webdriver
import selenium
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

import tensorflow as tf
import threading
import multiprocessing

import time


def download(yea):

    driver = webdriver.Chrome()

    years = [yea]

    for year in years:

        website = 'https://www.imf.org/external/np/fin/ert/GUI/Pages/CountryDataBase.aspx'
        driver.get(website)

        # date range

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_RadioDateRange")))

        driver.find_element_by_id(
            'ctl00_ContentPlaceHolder1_RadioDateRange').click()

        # from month
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_SelectFromMonth")))

        select = Select(
            driver.find_element_by_id(
                'ctl00_ContentPlaceHolder1_SelectFromMonth'))
        select.select_by_value('1')

        # from day
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_SelectFromDay")))

        select = Select(
            driver.find_element_by_id(
                'ctl00_ContentPlaceHolder1_SelectFromDay'))
        select.select_by_value('1')

        # from year
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_SelectFromYear")))

        select = Select(
            driver.find_element_by_id(
                'ctl00_ContentPlaceHolder1_SelectFromYear'))
        select.select_by_value('{}'.format(year))

        # to month
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_SelectToMonth")))

        select = Select(
            driver.find_element_by_id(
                'ctl00_ContentPlaceHolder1_SelectToMonth'))
        select.select_by_value('1')

        # to day
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_SelectToDay")))

        select = Select(
            driver.find_element_by_id('ctl00_ContentPlaceHolder1_SelectToDay'))
        select.select_by_value('1')

        # to year
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_SelectToYear")))

        select = Select(
            driver.find_element_by_id(
                'ctl00_ContentPlaceHolder1_SelectToYear'))
        select.select_by_value('{}'.format(year + 1))

        # continue

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_BtnContinue")))

        driver.find_element_by_id(
            'ctl00_ContentPlaceHolder1_BtnContinue').click()

        # select all

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_BtnSelect")))

        driver.find_element_by_id(
            'ctl00_ContentPlaceHolder1_BtnSelect').click()

        # continue

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_BtnContinue")))

        driver.find_element_by_id(
            'ctl00_ContentPlaceHolder1_BtnContinue').click()

        # prepare report

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "ctl00_ContentPlaceHolder1_imgBtnPrepareReport")))

        driver.find_element_by_id(
            'ctl00_ContentPlaceHolder1_imgBtnPrepareReport').click()

        link = driver.find_element_by_link_text('TSV')

        link.click()
        print('{} downloaded'.format(year))
        time.sleep(10)

    driver.stop_client()
    driver.close()


if __name__ == "__main__":

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        worker_threads = []

        years = list(range(1994, 2020, 1))

        for i in range(0, len(years), 4):

            for year in years[i:min(i + 4, len(years))]:

                def worker_fn():
                    return download(year)

                t = threading.Thread(target=worker_fn)
                t.start()
                worker_threads.append(t)

            # Wait for all workers to finish
            coord.join(worker_threads)
