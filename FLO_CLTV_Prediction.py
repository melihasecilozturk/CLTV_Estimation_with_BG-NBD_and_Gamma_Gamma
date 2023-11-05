##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##############################################################

# Business Problem
###############################################################
# FLO wants to determine a roadmap for sales and marketing activities.
# In order for the company to make medium-long term plans, it is necessary to estimate the potential value that
# existing customers will provide to the company in the future.

###############################################################


# The data set consists of information obtained from the past shopping behavior of customers who made their last purchases
# via OmniChannel (both online and offline shopping) in 2020 - 2021.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, iOS, Desktop, Mobile, Offline)
# last_order_channel : Channel where last purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_date : Customer's last purchase date
# last_order_date_online : The customer's last purchase date on the online platform
# last_order_date_offline : The last shopping date of the customer on the offline platform
# order_num_total_ever_online : Total number of purchases made by the customer on the online platform   F
# order_num_total_ever_offline : Total number of purchases made by the customer offline   F
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases  M
# customer_value_total_ever_online : Total fee paid by the customer for online shopping  M
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months

###############################################################
###############################################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
# Data Preparation

df_ = pd.read_csv("/Users/melihasecilozturk/Desktop/miuul/ödevler/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()


df.describe().T
#Define the outlier_thresholds and replace_with_thresholds functions required to suppress outliers.df
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable) #üstte yazdığımız fonksiyonu çağırdık
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

# Note: When calculating cltv, frequency values must be integer. Therefore, round the lower and upper limits with round().
# Suppress the "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_
# value_total_ever_online" variables if they have outliers.

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
               "customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

# Omnichannel means that customers shop both online and offline platforms.
# Create new variables for each customer's total number of purchases and spending.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Examine variable types. Change the type of variables expressing date to date.
# first way
df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
df["last_order_date_offline"]= df["last_order_date_offline"].astype("datetime64[ns]")

# second way
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df.dtypes

# Creating the CLTV Data Structure
# Take 2 days after the date of the last purchase in the data set as the analysis date.
df["last_order_date"].max()
analysis_date = dt.datetime(2021,6,1)

# Create a new cltv dataframe containing customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.

# recency: Time since last purchase. Weekly. (user specific)
# T: Age of the customer. Weekly. (how long before the date of analysis was the first purchase made)
# frequency: total number of recurring purchases (frequency>1)
# monetary: average earnings per purchase

#create an empty cltv_df dataframe
cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) // 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))//7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]


cltv_df.head()

#  Monetary value will be expressed as the average value per purchase, and recency and tenure values will be expressed in weekly basis.
###############################################################
# BG/NBD, Establishing Gamma-Gamma Models, Calculating 6-month CLTV
###############################################################

# Create BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Estimate the expected purchases from customers within 3 months and add it to the cltv dataframe as exp_sales_3_month.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# Estimate the expected purchases from customers within 6 months and add it to the cltv dataframe as exp_sales_6_month.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# Take a look at the 10 people who will make the most purchases in the 3rd and 6th months.
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]
cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

# Fit the Gamma-Gamma model. Estimate the average value that customers will leave and add it to the cltv dataframe as exp_average_value.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df.dtypes
df.dtypes

# Calculate 6-month CLTV and add it to the dataframe with the name cltv.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

# Observe the 20 people with the highest CLTV values.
cltv_df.sort_values("cltv",ascending=False)[:20]



# Creating Segments Based on CLTV

# Divide all your customers into 4 groups (segments) according to 6-month cltv and add the group names to the data set.
# Add it to the dataframe with the name cltv_segment.


cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()



# Make short 6-month action suggestions to the management for 2 groups you choose among 4 groups.

cltv_df.sort_values(by="cltv", ascending=False).head(50)

#I chose C and D. If discounts, campaigns, etc. are offered to this group, whose value is relatively lower, they may shop.

# BONUS: Functionalize the entire process.

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

# We created a treshhold. We assigned lower limit and upper limit values
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

df_ = pd.read_csv("/Users/melihasecilozturk/Desktop/miuul/ödevler/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()


def create_cltv_pp(dataframe, month=6):
    # 1. Veri Ön İşleme

    replace_with_thresholds(df, "order_num_total_ever_online")
    replace_with_thresholds(df, "order_num_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_online")


    df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
    df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
    df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
    df["last_order_date_offline"]= df["last_order_date_offline"].astype("datetime64[ns]")

    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    analysis_date = dt.datetime(2021,6,1)

    cltv_df = pd.DataFrame()

    cltv_df["customer_id"] = df["master_id"]
    cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) // 7
    cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))//7
    cltv_df["frequency"] = df["order_num_total"]
    cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]


#    cltv_df = dataframe.groupby('Customer ID').agg(
#        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
#                        lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
#        'Invoice': lambda Invoice: Invoice.nunique(),
#         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

#    cltv_df.columns = cltv_df.columns.droplevel(0)
#    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
#    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
#    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
#    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

# Estimate the expected purchases from customers within 3 months and add it to the cltv dataframe as exp_sales_3_month.
    cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# Estimate the expected purchases from customers within 6 months and add it to the cltv dataframe as exp_sales_6_month.
    cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])




# Creating GAMMA-GAMMA Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

# Calculation of CLTV with BG-NBD and GG model
    cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
    cltv_df["cltv"] = cltv
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df


df = df_.copy()

cltv_final = create_cltv_pp(df)

cltv_final.to_csv("cltv_prediction_flo.csv")











