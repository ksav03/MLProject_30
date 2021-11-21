# Import data
import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\\kesha\\Desktop\\Machine_Learning\\Project\\dublinbikes_20180701_20181001.csv")


'''
Function - extract_data
Parameters - file, station_id
Function takes csv file pointer, station_id, converts dataframe to np.array
and saves/returns new file with required data for given station.
Original File Header - Station Id, time, last update, name, bike stands, available bike stands,
available bikes, status, address, latitude, longitude
'''
def extract_data(file, ID):
    
    station_id = np.array(df.iloc[:,0])

    # Times at which API collects data from station, regural interval of 5mins
    time = np.array(df.iloc[:,1])
    
    # total number of stands at the station, this is a constant value for each station
    num_stand = np.array(df.iloc[:,4]) 
    
    # number of avaible bikes at the station
    num_bikes = np.array(df.iloc[:,6]) 

    #data = np.column_stack((((station_id, time, num_stand, num_bikes))))
    data = np.column_stack(((station_id, num_stand, num_bikes)))
    
    print("Time at index 0",time[0])
    print("Time at last index", time[time.size-1])
    
    # return the indexes that get a match as tuple datatype
    x = (np.where(station_id == ID))[0]
    
    print("total no. of data", station_id.size)
    print("total no. of data matching ID number", x.size)
    
    # Save the extracted data to a new file 
    num_column = 5
    extracted_data = np.zeros((x.size, num_column))
    #time_col = np.zeros((x.size,1),dtype=str)
    
    precision = 3 # precision for decimal numbers 
    #threshold = round(2/num_stand[0],precision)
    threshold = 2 # this is the value that distinguishes the label
    
    # x contains all the indexes that matches desired station id, note it contains for the entire quarter
    counter = 0
    for index in x:
        extracted_data[counter,0:3] = data[index,:]
        
        # ratio of num of available bikes at the stand to totalnumber of stand at the station 
        # ratio gives what percentage of bikes are availalble at the station
        ratio =  round(extracted_data[counter,2] / num_stand[0], precision)
        extracted_data[counter,3] = ratio    
        
        # if more than 2 bikes are at the station no bikes needed, label = 
        if (extracted_data[counter,2] <= threshold): 
            extracted_data[counter,4] = 1
        elif extracted_data[counter,2] >= extracted_data[counter,1] -threshold:
            extracted_data[counter,4] = -1
        else:
            extracted_data[counter,4] = 0
        
        counter += 1

    print(extracted_data)
    dataset = pd.DataFrame({'Station_ID':extracted_data[:,0], 'Total Stand':extracted_data[:,1],
                            'Available Bikes':extracted_data[:,2], 'Ratio BikeVsStand':extracted_data[:,3],
                            'Label':extracted_data[:,4]})
    print(dataset.head)
    
    # Write a new csv file 
    dataset.to_csv("New_file.csv", encoding='utf-8',index=False)
       
extract_data(df, 2)
    


    
