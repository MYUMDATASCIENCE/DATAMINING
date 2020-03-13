#WQD7005 - DATA MINING
#INSTRUCTOR : PROF. DR. TEH YING WAH!
#ZAIMIE AZMIN BIN ZAINUL ABIDIN (OLD MATRIX : WQD190018 /NEW MATRIX : 17202336)
#ASSIGNMENT TITLE : CRUDE OIL WTI AND BRENT PRICE PREDICTION
#MILESTONE 1 : WEB SCRAPING
#Web crawling the real time data by using Python  (WQD7004 Programming for Data Science)
#--------------------------------------

#IMPORT PACKAGE BEUTIFULSOUP & SELENIUM
from bs4 import BeautifulSoup as soup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

browser = webdriver.Chrome("/Users/mac/Documents/webdriver/chromedriver")

#URL PAGE SOURCE
url = "https://markets.businessinsider.com/commodities/historical-prices/oil-price/usd?type=brent"

#OPEN BROWSER WITH PAGE URL
browser.get(url)

#SELECT DAY
elem_from_day = Select(browser.find_element_by_id("historic-prices-start-day"))
elem_from_day.select_by_index(0)

#SELECT MONTH
elem_from_month = Select(browser.find_element_by_id("historic-prices-start-month"))
elem_from_month.select_by_index(0)

#SELECT YEAR
elem_from_year = Select(browser.find_element_by_id("historic-prices-start-year"))
elem_from_year.select_by_index(0)

#SHOW PRICE
elem_show = browser.find_element_by_id("request-historic-price")
elem_show.click()

#WAITING PAGE TO LOAD
try:
  myElem = WebDriverWait(browser, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'header-row')))
except:
    print("Loading....")

#GET HTML FROM PAGE SOURCE
browser.execute_script("window.scrollTo(0,100)")
soap_html = soup(browser.page_source, "html.parser")
rowcount=0

#OPEN CSV FILE
filename="mywebscrapBrentFinal.csv"
f=open(filename,"w")

#WRITE TABLE HEADERS
for tr in soap_html.find_all("tr", {"class": "header-row"}):
    tds = tr.find_all('th')
    mydate = tds[0].text.strip()
    myclosingprice = tds[1].text.strip()
    myopen = tds[2].text.strip()
    mydailyhigh = tds[3].text.strip()
    mydailylow = tds[4].text.strip()
    f.write(mydate + "," + myclosingprice + "," + myopen + "," + mydailyhigh + "," + mydailylow + "\n")

#WRITE TABLE ROWS
for tr in soap_html.find_all('tr')[2:]:
    tds = tr.find_all('td')
    mydate=tds[0].text.strip()
    myclosingprice=tds[1].text.strip()
    myopen=tds[2].text.strip()
    mydailyhigh=tds[3].text.strip()
    mydailylow=tds[4].text.strip()
    f.write(mydate + "," + myclosingprice + "," + myopen + "," + mydailyhigh + "," + mydailylow + "\n")
    rowcount=rowcount+1
    print("Writing to file...row = ",rowcount)

#CLOSE FILE
f.close

#PRINT TOTAL ROWS WRITE TO FILE
print("Total Rows = ",rowcount)

#CLOSE BROWSER
browser.close()