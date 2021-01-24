import pyautogui as pg
import webbrowser
from selenium import webdriver

browser = webdriver.Safari()
#browser.maximize_window()

listOfFoods = ["carrot", "apple", "strawberries"]
prices = []
##pg.sleep(2)
for food in listOfFoods:
    browser.get("https://shop.gianteagle.com/west-seventh-street/search?q=" + food)
    bodyText = browser.page_source
    #print(bodyText)
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
        avg = total/numProduct
        prices.append(avg)
        print(total/numProducts)
            
    except:
        print('Was not able to find product in grocery store')
