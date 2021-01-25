
from product_info_get import WebScrape

class UserData:

    def __init__(self, name):
        self.name = name
        self.freq_data = {}
        self.freq_data[1] = set()
        self.list_current = []
        self.price_getter = WebScrape()

    def list_reset(self):
        self.list_current = []

    def find_freqkey(self, product):
        key = None
        for k in self.freq_data.keys():
            if product in self.freq_data[k]:
                key = k
                break

        return key

    def is_recommended_product(self, product):
        return self.find_freqkey() > 0

    def freq_upgrade(self, product, freq_prev):
        self.freq_data[freq_prev].remove(product)

        if freq_prev + 1 in self.freq_data.keys():
            self.freq_data[freq_prev + 1].add(product)
        else:
            self.freq_data[freq_prev + 1] = set()
            self.freq_data[freq_prev + 1].add(product)

    def add_product(self, product):
        self.list_current.append(product)
        product_freqkey = self.find_freqkey(product)
        if not product_freqkey:
            self.freq_data[1].add(product)
        else:
            self.freq_upgrade(product, product_freqkey)

    def get_ordered_recs_prices(self):
        recs = []
        for freq in sorted(list(self.freq_data.keys()), reverse = True):
            recs.extend(self.freq_data[freq])

        price = self.price_getter.selenium_script(recs)
        return recs, price

    def debug_print(self):
        print(self.freq_data, self.list_current)



