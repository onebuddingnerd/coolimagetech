import pyautogui as pg
import webbrowser
from selenium import webdriver

class WebScrape(object):
    def __init__(self, listOfFoods):
        self.listOfFoods = listOfFoods
        
    def selenium_script(self):
        browser = webdriver.Safari()
        #browser.maximize_window()

        totalPrice = 0
        length = len(self.listOfFoods)
        for food in self.listOfFoods:
            browser.get("https://shop.gianteagle.com/west-seventh-street/search?q=" + food)
            #bodyText = browser.page_source
            pg.sleep(4)
            try:
                name = browser.find_element_by_id('content')
                total = 0
                numProducts = 0
                
                for item in name.text.split():
                    if "$" in item:
                        dollar_sign = item.index("$")
                        addToPrices = True
                        for char in item[dollar_sign+1:]:
                            if not (ord(char) == 46 or (48 <= ord(char) and ord(char) <= 57)):
                                addToPrices = False
                        if addToPrices:
                            add = float(item[dollar_sign+1:])
                            if add < 50:
                                total += add
                                numProducts += 1
                avg = total/numProducts
                totalPrice += avg
                        
            except:
                print('Was not able to find product in grocery store')

        #print(prices)
        return totalPrice/length

webObj = WebScrape(['orange', 'strawberries', 'potato'])
print(webObj.selenium_script())
