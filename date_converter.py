import pandas as pd
import datetime


def time_to_seconds(data):
    """
    This function will return the total seconds passed during 24 time
    input data: HH:MM:SS
    return: Seconds
    """
    seconds_value = []
    month_value = []
    day_value = []
    for dates in data:
        # | Get the time values, date is also stored if needed
        date, separation, time = dates.partition(' ')
        hour, minute, second = time.split(':')
        # | Convert into seconds and store in array
        total_time_seconds = int(hour) * 3600 + int(minute) * 60 + int(second)
        seconds_value.append(total_time_seconds)
        #| Format the dates
        date_info = datetime.datetime.strptime(date, '%Y-%m-%d')
        month_value.append(date_info.month)
        day_value.append(date_info.day)

    return seconds_value, month_value, day_value


def time_to_minutes(data):
    """
    This function will return the total minutes passed during 24 time
    input data: HH:MM:SS
    return: Minutes
    """
    minute_value = []
    month_value = []
    day_value = []
    for dates in data:
        # | Get the time values, date is also stored if needed
        date, separation, time = dates.partition(' ')
        hour, minute, second = time.split(':')
        # | Convert into minutes and store in array
        total_time_minutes = int(hour) * 60 + int(minute)
        minute_value.append(total_time_minutes)
        # | Format the dates
        date_info = datetime.datetime.strptime(date, '%Y-%m-%d')
        month_value.append(date_info.month)
        day_value.append(date_info.day)

    return minute_value, month_value, day_value


csv_path = "dataset\labelled_dataset_2.csv"
df = pd.read_csv(csv_path)

data = df.iloc[:1000, 2]
print(f'[INFO] Data type is currently:\n{data[1]}')
print(f'[INFO] Converting into seconds')

minutes, months, days = time_to_minutes(data)
print(f'[INFO] Minute data: \n{minutes[0:30]}')
print(f'[INFO] Month data: \n{months[0:30]}')
print(f'[INFO] Day data: \n{days[0:30]}')
# print(time_to_seconds.__doc__)

