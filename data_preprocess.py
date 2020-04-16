#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 05 17:40:38 2020
@author: devendraswami
"""

import numpy as np
import pandas as pd
# import googletrans
import sys
import requests
pd.set_option("display.max_columns", None)


def translate(x, key, src='ru', dest='en'):
    """
    Computes the content cost

    Parameters
    ------------
    x : pd.Series of series to be translated 
    key : translation key
    src: Russian Language ('ru')
    dest: English Language ('en')

    Returns
    ---------
    trans_list : list containing translation of items in x
    """
    url = 'https://translate.yandex.net/api/v1.5/tr.json/translate'
    params = dict(
        key=key,
        lang=src+'-'+dest
    )
    dlen = len(x)
    counter = 0
    translated_text = []
    while dlen > 0:
        payload = {'text': x[counter:counter+100]}
        response = requests.post(url=url, params=params, data=payload)
        try:
            result = response.json()['text']
#             print(result)
            translated_text.extend(result)
            dlen -= 100
            counter += 100
        except:
            print(response)
            sys.exit(1)
        print("Current progress: ", np.round(
            min(counter, len(x))/len(x)*100, 2), "%")
#     print(translated_text)
    dictionary = dict(zip(x, translated_text))
    trans_list = [dictionary.get(item, item) for item in x]
    return trans_list


def build_master_df():
    """
    Import all datasets, translate, merge into a dataframe and dump into master_df.csv
    Call this function if master_df.csv is not present in your folder since it is a time taking step
    """

    # ['date', 'date_block_num', 'shop_id', 'item_id', 'item_price','item_cnt_day']
    sales = pd.read_csv("sales_train.csv")
    shops = pd.read_csv("shops.csv")        # ['shop_name', 'shop_id']
    # ['item_name', 'item_id', 'item_category_id']
    items = pd.read_csv("items.csv")
    # ['item_category_name', 'item_category_id']
    item_categories = pd.read_csv("item_categories.csv")

    master_df = pd.merge(sales, shops, on="shop_id")
    master_df = pd.merge(master_df, items, on="item_id")
    master_df = pd.merge(master_df, item_categories, on="item_category_id")

    master_df['shop_name_en'] = translate(
        master_df['shop_name'], "trnsl.1.1.xxxxxxxxxxxxxxxxx")
    master_df['item_name_en'] = translate(
        master_df['item_name'], "trnsl.1.1.xxxxxxxxxxxxxxxxx")
    master_df['item_category_name_en'] = translate(
        master_df['item_category_name'], "trnsl.1.1.xxxxxxxxxxxxxxxxx")
    master_df = master_df.drop(
        labels=['shop_name', 'item_name', 'item_category_name'], axis=1)
    master_df = master_df.rename(columns={'shop_name_en': 'shop_name',
                                          'item_name_en': 'item_name',
                                          'item_category_name_en': 'item_category_name'})

    master_df.to_csv('master_df.csv', sep=',', index=False, header=True)


def data_gathering():
    """
    Gathers all data from external sources (DAILY VIEW)

    Returns
    ---------
    df : output pandas dataframe containing following columns
    ['date', 'date_block_num' i.e. month number, 'shop_id', 'item_id', 'item_price',
       'item_cnt_day', 'shop_name', 'item_category_id', 'item_name',
       'item_category_name', 'ID',
       'isMovie', 'isMusic', 'isBook', 'isGame', 'isGift', 'isAccessory', 
       'isProgram', 'isPaymentCard', 'isService, 'isDelivery', 'isBatteries']'
    """

    df = pd.read_csv("master_df.csv")
    # Super Categories
    # ['Movie', 'Music','Accessories', 'Games', 'Gifts,','Payment Card', 'Program',
    #     'Books', 'Batteries', 'Service', 'Delivery']
    df["isMovie"] = df.apply(lambda row: int(
        "movie" in str(row["item_category_name"]).lower()), axis=1)
    df["isMusic"] = df.apply(lambda row: int(
        "music" in str(row["item_category_name"]).lower()), axis=1)
    df["isBook"] = df.apply(lambda row: int(
        "book" in str(row["item_category_name"]).lower()), axis=1)
    df["isGame"] = df.apply(lambda row: int("game" in str(row["item_category_name"]).lower() or
                                            "gaming" in str(row["item_category_name"]).lower()), axis=1)
    df["isGift"] = df.apply(lambda row: int(
        "gift" in str(row["item_category_name"]).lower()), axis=1)
    df["isAccessory"] = df.apply(lambda row: int(
        "accessories" in str(row["item_category_name"]).lower()), axis=1)
    df["isProgram"] = df.apply(lambda row: int(
        "program" in str(row["item_category_name"]).lower()), axis=1)
    df["isPaymentCard"] = df.apply(lambda row: int(
        "payment card" in str(row["item_category_name"]).lower()), axis=1)
    df["isService"] = df.apply(lambda row: int(
        "service" in str(row["item_category_name"]).lower()), axis=1)
    df["isDelivery"] = df.apply(lambda row: int(
        "delivery" in str(row["item_category_name"]).lower()), axis=1)
    df["isBatteries"] = df.apply(lambda row: int(
        "batteries" in str(row["item_category_name"]).lower()), axis=1)
    return df


def add_data(df, filter_field, field, min_val, max_val):
    """
    Transpose the data in field column as per min_val and max_val of filter_field col

    Appends the extra columns to the passed dataframe
    """
    def fill_data(row, i):
        if row[filter_field] == i:
            return row[field]
        else:
            return None
    for i in range(min_val, max_val+1, 1):
        df[field+str(i)] = df.apply(lambda row: fill_data(row, i), axis=1)


def data_preprocess():
    """
    Preprocess the data so that all information of every unique combination 
    of shop and item is in single row.
    Also dump the output dataframe in preprocess_df.csv

    Returns
    ---------
    data_df : output pandas dataframe
    """
    data = data_gathering()

    count_df = data.filter(items=['shop_id', 'item_id', 'date_block_num', 'item_cnt_day'], axis=1).groupby(
        ['shop_id', 'item_id', 'date_block_num']).sum()
    count_df.reset_index(inplace=True)
    count_df.rename(columns={'item_cnt_day': 'item_cnt_mnth'}, inplace=True)
    add_data(count_df, "date_block_num", "item_cnt_mnth", 0, 33)
    count_df = count_df.drop(labels=["date_block_num", "item_cnt_mnth"], axis=1).groupby(
        ['shop_id', 'item_id']).sum()
    count_df.reset_index(inplace=True)

    other_df = data.drop(labels="item_cnt_day", axis=1).groupby(
        ['shop_id', 'item_id', 'date_block_num']).mean()
    other_df.reset_index(inplace=True)
    add_data(other_df, "date_block_num", "item_price", 0, 33)
    other_df = other_df.drop(labels=["date_block_num", "item_price"], axis=1).groupby(
        ['shop_id', 'item_id']).mean()

    data_df = pd.merge(count_df, other_df, on=['shop_id', 'item_id'])
    data_df.to_csv('preprocess_df.csv', sep=',', index=False, header=True)

    return data_df


def train_data(period=12, min_time=12, max_time=33):
    """
    Parameters
    ------------
    period: how many data of past months is needed to predict next mnths data 
    min_time: index of first month data to be considered as label in training
    max_time: index of last month data to be considered as label in training

    Returns
    ---------
    train_df: training pandas dataframe (contain NA for mnths with no item count)
    """

    cols = ['shop_id', 'item_id', 'item_category_id', 'isMovie', 'isMusic', 'isBook', 'isGame', 'isGift',
            'isAccessory', 'isProgram', 'isPaymentCard', 'isService', 'isDelivery', 'isBatteries', 'label']

    for i in range(period):
        cols.append('item_cnt_mnth'+str(i))
        cols.append('item_price'+str(i))

    train_df = pd.DataFrame(columns=cols)
    df = pd.read_csv("preprocess_df.csv")

    for i in range(max_time - min_time + 1):
        temp = df.filter(items=['shop_id', 'item_id', 'item_category_id', 'isMovie',
                                'isMusic', 'isBook', 'isGame', 'isGift', 'isAccessory', 'isProgram',
                                'isPaymentCard', 'isService', 'isDelivery', 'isBatteries'])
        for j in range(period):
            temp['item_cnt_mnth'+str(j)] = df['item_cnt_mnth'+str(i+j)]
            temp['item_price'+str(j)] = df['item_price'+str(i+j)]
        temp["label"] = df['item_cnt_mnth'+str(i+period)]
        train_df = train_df.append(temp, ignore_index=True, sort=False)

    cols = []
    for i in range(period):
        cols.append('item_cnt_mnth'+str(i))
    train_df[cols] = train_df[cols].replace({0: np.nan})
    return train_df


def test_data(period=12, max_time=33):
    """
    Parameters
    ------------
    period: how many data of past months is needed to predict next mnths data 
    max_time: index of last month data to be considered as x_data in testing

    Returns
    ---------
    test_df : testing pandas dataframe
    """

    df = pd.read_csv("preprocess_df.csv")
    test_df = df.filter(items=['shop_id', 'item_id', 'item_category_id', 'isMovie',
                               'isMusic', 'isBook', 'isGame', 'isGift', 'isAccessory', 'isProgram',
                               'isPaymentCard', 'isService', 'isDelivery', 'isBatteries'])

    for j in range(period):
        test_df['item_cnt_mnth' +
                str(j)] = df['item_cnt_mnth'+str(max_time - period + j + 1)]
        test_df['item_price'+str(j)] = df['item_price' +
                                          str(max_time - period + j + 1)]

    test = pd.read_csv("test.csv")
    test_df = pd.merge(test_df, test, on=["shop_id", "item_id"])
    return test_df
