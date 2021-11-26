import sys
import numpy as np
import pandas as pd

from collections import Counter
from datetime import datetime

df = pd.read_csv("dublinbikes_20210401_20210701.csv")

'''
Function - extract_data
Parameters - file, station_id
Function takes csv file pointer, station_id, converts dataframe to np.array
and saves/returns new file with required data for given station.
Original File Header - Station Id, time, last update, name, bike stands, available bike stands,
available bikes, status, address, latitude, longitude
'''


def time_converter(time_arr):
    time_obj_arr, minute_arr, weekday_arr, week_arr = [], [], [], []
    for temp_str in time_arr:
        datetime_object = datetime.strptime(temp_str, '%Y-%m-%d %H:%M:%S')
        time_obj_arr.append(datetime_object)
        weekday_arr.append(datetime_object.weekday() + 1)
        minute_arr.append(datetime_object.minute + datetime_object.hour * 60)
        week_arr.append(datetime_object.isocalendar()[1])
    return time_obj_arr, weekday_arr, minute_arr, week_arr


def read_file(file):
    # Discard colum 2, 3, 7, 8, 9, 10
    station_id = np.array(df.iloc[:, 0])
    simp_counter = Counter(station_id).values()
    print(f'\nTotal number of station: {len(simp_counter)}\n'
          f'Average number of data for each station: {round(sum(simp_counter) / len(simp_counter), 1)}\n'
          f'Total number of data: {len(station_id)}')
    # Times at which API collects data from station, regural interval of 5mins
    time_point = np.array(df.iloc[:, 1])
    _, weekday_array, minute_array, week_array = time_converter(time_point)

    # total number of stands at the station, this is a constant value for each station
    num_stand = np.array(df.iloc[:, 4])
    avai_stands = np.array(df.iloc[:, 5])
    avai_bikes = np.array(df.iloc[:, 6])  # number of avaible bikes at the station
    data_ = np.column_stack((station_id, num_stand, week_array, weekday_array, minute_array, avai_stands, avai_bikes))
    return data_


def extract_data(data_, id_):
    # return the indexes that get a match as tuple datatype
    x = (np.where(data_[:, 0] == id_))[0]
    print(f"\n\nTotal no. of data matching ID {id_}: {x.size}")

    # Save the extracted data to a new file
    num_column = 6 + 1 + 1  # 5 Features, 1 ratio, 1 label (3 classes -1, 0, 1)
    extracted_data = np.zeros((x.size, num_column))
    week_int, weekday_int, minute_int = [], [], []
    precision = 3  # precision for decimal numbers
    # threshold = round(2/num_stand[0],precision)
    threshold = 0.15  # this is the value that distinguishes the label

    # x contains all the indexes that matches desired station id, note it contains for the entire quarter
    counter = 0
    for index in x:
        extracted_data[counter, 0] = data_[index, 0]
        extracted_data[counter, 1] = data_[index, 1]
        week_int.append(data_[index, 2])
        weekday_int.append(data_[index, 3])
        minute_int.append(data_[index, 4])
        extracted_data[counter, 3] = data_[index, 5]
        extracted_data[counter, 4] = data_[index, 6]

        # ratio of num of available bikes at the stand to total number of stand at the station
        # ratio gives what percentage of bikes are availalble at the station
        ratio = round(extracted_data[counter, 4] / extracted_data[counter, 1], precision)
        extracted_data[counter, 5] = ratio

        # if more than 2 bikes are at the station no bikes needed, label =
        if extracted_data[counter, 5] <= threshold:                 # Below 15%: Label as 1 for adding
            extracted_data[counter - 6:counter + 1, 6] = 1
        elif extracted_data[counter, 5] >= (1 - threshold):         # Over 85%: Label as 2 for removing bikes
            extracted_data[counter - 6:counter + 1, 6] = 2
        else:
            extracted_data[counter, 6] = 0

        counter += 1

    dataset = pd.DataFrame({'Station_ID': extracted_data[:, 0],
                            'Total Stand': extracted_data[:, 1],
                            'Week': week_int,
                            'Weekday': weekday_int,
                            'Minute' : minute_int,
                            'Available Stands': extracted_data[:, 3],
                            'Available Bikes': extracted_data[:, 4],
                            'Ratio BikeVsStand': extracted_data[:, 5],
                            'Label': extracted_data[:, 6]})

    # Write a new csv file
    dataset.to_csv(f"dataset/labelled_dataset_{id_}.csv", encoding='utf-8', index=False)


def main(args):
    data = read_file(df)
    for id_ in range(2, 118):
        extract_data(data, id_)


if __name__ == '__main__':
    main(sys.argv)

