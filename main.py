import numpy as np
import pandas as pd

hr = pd.read_csv('CoSupplyChain/DataCoSupplyChainDataset.csv')
hr.info()
hrs = hr.select_dtypes('number')
hrs.info()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
