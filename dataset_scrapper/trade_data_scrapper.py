from selenium import webdriver
import selenium
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException



driver = webdriver.Chrome()

# countries = ['004','577','008','012','636','016','020','024','660','028','899','032','051','533','946','036','040','031','044','048','050','052','112','056','084','204','060','064','068','535','070','072','074','076','080','086','092','096','100','854','108','132','116','120','124','129','136','471','140','148','152','156','162','166','170','174','178','180','184','188','384','191','192','531','196','203','208','262','212','214','221','218','818','222','226','232','233','748','231','697','568','492','238','234','242','246','251','838','258','260','266','270','268','276','288','292','300','304','308','316','320','324','624','328','332','340','344','348','352','699','360','364','368','372','376','381','388','392','400','398','404','296','408','410','414','417','473','418','428','422','426','430','434','440','442','446','807','450','454','458','462','466','470','584','478','480','175','484','583','498','496','499','500','504','508','104','516','520','524','528','530','536','540','554','558','562','566','570','574','290','637','580','579','527','512','586','585','275','591','598','600','604','608','612','616','620','634','642','643','646','654','659','662','670','882','678','682','686','688','891','690','837','694','702','534','703','705','090','706','710','728','724','839','144','666','729','736','740','752','757','760','490','762','834','999','764','626','768','772','776','780','788','792','795','796','798','800','804','784','826','849','858','860','548','862','704','876','879','732','887','894','716']

countries = ['100','854','108','132','116','120','124','129','136','471','140','148','152','156','162','166','170','174','178','180','184','188','384','191','192','531','196','203','208','262','212','214','221','218','818','222','226','232','233','748','231','697','568','492','238','234','242','246','251','838','258','260','266','270','268','276','288','292','300','304','308','316','320','324','624','328','332','340','344','348','352','699','360','364','368','372','376','381','388','392','400','398','404','296','408','410','414','417','473','418','428','422','426','430','434','440','442','446','807','450','454','458','462','466','470','584','478','480','175','484','583','498','496','499','500','504','508','104','516','520','524','528','530','536','540','554','558','562','566','570','574','290','637','580','579','527','512','586','585','275','591','598','600','604','608','612','616','620','634','642','643','646','654','659','662','670','882','678','682','686','688','891','690','837','694','702','534','703','705','090','706','710','728','724','839','144','666','729','736','740','752','757','760','490','762','834','999','764','626','768','772','776','780','788','792','795','796','798','800','804','784','826','849','858','860','548','862','704','876','879','732','887','894','716']

for country in countries:

    website = 'https://www.trademap.org/tradestat/Index.aspx'
    driver.get(website)

    # input product types

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.ID, "ctl00_PageContent_RadComboBox_Product_Input")))

    driver.find_element_by_id(
        'ctl00_PageContent_RadComboBox_Product_Input').click()

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.ID, "ctl00_PageContent_RadComboBox_Product_c0")))

    driver.find_element_by_id(
        'ctl00_PageContent_RadComboBox_Product_c0').click()

    # input country

    driver.find_element_by_id(
        'ctl00_PageContent_RadComboBox_Country_Input').click()

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.ID, "ctl00_PageContent_RadComboBox_Country_c0")))

    driver.find_element_by_id(
        'ctl00_PageContent_RadComboBox_Country_c0').click()

    # click monthly
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.ID, "ctl00_PageContent_Button_TimeSeries_M")))

    driver.find_element_by_id('ctl00_PageContent_Button_TimeSeries_M').click()

    # trade balance
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.ID, "ctl00_NavigationControl_DropDownList_TradeType")))

    select = Select(
        driver.find_element_by_id(
            'ctl00_NavigationControl_DropDownList_TradeType'))
    select.select_by_value('B')

    # new country

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.ID, "ctl00_NavigationControl_DropDownList_Country")))

    select = Select(
        driver.find_element_by_id(
            'ctl00_NavigationControl_DropDownList_Country'))
    select.select_by_value(country)

    # rows per page

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.ID,
             "ctl00_PageContent_GridViewPanelControl_DropDownList_PageSize")))

    select = Select(
        driver.find_element_by_id(
            'ctl00_PageContent_GridViewPanelControl_DropDownList_PageSize'))
    select.select_by_value('300')

    # months per page

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.ID,
             "ctl00_PageContent_GridViewPanelControl_DropDownList_NumTimePeriod")))

    select = Select(
        driver.find_element_by_id(
            'ctl00_PageContent_GridViewPanelControl_DropDownList_NumTimePeriod'))
    select.select_by_value('20')

    

    years = set([])

    while True:

        while True:

            links = driver.find_elements_by_partial_link_text(
                'Balance in value in ')

            years_new = set([link.text for link in links])

            if years_new != years:
                years = years_new
                break

            if len(years) == 0:
                break

        if len(years) == 0:
            break

        # download

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID,
                 "ctl00_PageContent_GridViewPanelControl_ImageButton_Text")))

        print('dowloading')
        driver.find_element_by_id(
            'ctl00_PageContent_GridViewPanelControl_ImageButton_Text').click()

        # go to previous years

        button = driver.find_element_by_id(
            'ctl00_PageContent_GridViewPanelControl_ImageButton_Previous')
        
        disable = button.get_attribute('disabled')
        
        if disable is not None:
            break

        button.click()