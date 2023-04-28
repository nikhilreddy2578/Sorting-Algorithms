import csv
import time
import json
import pandas as pd
"""
Note : For test cases 7-10, you need to extract the required data (filter on conditions mentioned above)
and rename it to appropriate name as mentioned in the test case descriptions. You need to write the code
to perform this extraction and renaming, at the start of the skeleton file.
"""

column_names = ['tconst', 'primaryTitle', 'originalTitle', 'startYear',
                'runtimeMinutes', 'genres', 'averageRating', 'numVotes', 'ordering',
                'category', 'job', 'seasonNumber', 'episodeNumber', 'primaryName', 'birthYear',
                'deathYear', 'primaryProfession']


#############################################################################################################
# Data Filtering
#############################################################################################################
def data_filtering(filelocation, num):
    """
    Data Filtering is for the test cases from 7 to 10.
    filelocation: imdb_dataset.csv location
    num: if num == 1 -> filter data based on years (years in range 1941 to 1955)
         if num == 2 -> filter data based on genres (genres are either ‘Adventure’ or ‘Drama’)
         if num == 3 -> filter data based on primaryProfession (if primaryProfession column contains substrings
                        {‘assistant_director’, ‘casting_director’, ‘art_director’, ‘cinematographer’} )
         if num == 4 -> filter data based on primary Names which start with vowel character.

    """
    df = pd.read_csv(filelocation)  # Load the imdb_dataset.csv dataset
    if (num == 1):
        # NEED TO CODE
        # Implement your logic here for Filtering data based on years (years in range 1941 to 1955)
        # Store your filtered dataframe here
        df_year = df[(df['startYear'].between(1941, 1955, inclusive='both'))]
        df_year.reset_index(drop=True).to_csv("imdb_years_df.csv", index=False)

    if (num == 2):
        # NEED TO CODE
        # Implement your logic here for Filtering data based on genres (genres are either ‘Adventure’ or ‘Drama’)
        df_genres = df.loc[(df['genres']=="Adventure") | (df['genres']=="Drama")]  # Store your filtered dataframe here
        df_genres.reset_index(drop=True).to_csv(
            "imdb_genres_df.csv", index=False)
    if (num == 3):
        # NEED TO CODE
        # Implement your logic here for Filtering data based on primaryProfession (if primaryProfession column contains
        # substrings {‘assistant_director’, ‘casting_director’, '‘art_director’', ‘cinematographer’} )
        df_professions = df.loc[df["primaryProfession"].str.contains(('assistant_director|casting_director|art_director|cinematographer'))]  # Store your filtered dataframe here
        df_professions.reset_index(drop=True).to_csv(
            "imdb_professions_df.csv", index=False)
    if (num == 4):
        # NEED TO CODE
        # Implement your logic here for Filtering data based on primary Names which start with vowel character.
        # df_vowels = df[df['primaryTitle'].str.strip().str.startswith(('a','e','i','o','u','A','E','I','O','U'))]  # Store your filtered dataframe here
        # df_vowels.reset_index(drop=True).to_csv(
            # "imdb_vowel_names_df.csv", index=False)
        vowels = ['a', 'e', 'i', 'o', 'u']
        df_vowels = df[df['primaryName'].str.lower().str.strip().str[0].isin(vowels)]
        df_vowels.reset_index(drop=True).to_csv("imdb_vowel_names_df.csv",index=False)


#############################################################################################################
# Quick Sort
#############################################################################################################
def pivot_element(arr,low,high):
    # CODE For identifiying the pivot element
    pivot = arr[high]
    i = low - 1
    col_len=len(arr[0])
    for j in range(low, high):
        if type(arr[j][1])==str:
            if arr[j][1].strip() < pivot[1].strip():
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
            elif arr[j][1].strip()==pivot[1].strip() and col_len>2:
                for x in range(2,col_len):
                    if type(arr[j][x])==str:
                        if(arr[j][x].strip()<pivot[x].strip()):
                            i+=1
                            arr[i], arr[j] = arr[j], arr[i]
                            break
                        elif(arr[j][x].strip()>pivot[x].strip()):
                            break
                    else:
                        if(arr[j][x]<pivot[x]):
                            i+=1
                            arr[i], arr[j] = arr[j], arr[i]
                            break
                        elif(arr[j][x]>pivot[x]):
                            break                
        else:
            if arr[j][1] < pivot[1]:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
            elif arr[j][1]==pivot[1] and col_len>2:
                for x in range(2,col_len):
                    if type(arr[j][x])==str:
                        if(arr[j][x].strip()<pivot[x].strip()):
                            i+=1
                            arr[i], arr[j] = arr[j], arr[i]
                            break
                        elif(arr[j][x].strip()>pivot[x].strip()):
                            break
                    else:
                        if(arr[j][x]<pivot[x]):
                            i+=1
                            arr[i], arr[j] = arr[j], arr[i]
                            break
                        elif(arr[j][x]>pivot[x]):
                            break        
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i+1


def quicksort(arr, columns):
    """
    The function performs the QuickSort algorithm on a 2D array (list of lists), where
    the sorting is based on specific columns of the 2D array. The function takes two parameters:

    arr: a list of lists representing the 2D array to be sorted
    columns: a list of integers representing the columns to sort the 2D array on

    The function first checks if the length of the input array is less than or equal to 1,
    in which case it returns the array as is. Otherwise, it selects the middle element of
    the array as the pivot, and splits the array into three parts: left, middle, right.

    Finally, the function calls itself recursively on the left and right sub-arrays, concatenates
    the result of the recursive calls with the middle sub-array, and returns the final sorted 2D array.
    """
    if len(arr) <= 1 :
        return arr
    # NEED TO CODE
    # Implement Quick Sort Algorithm
    # return Sorted array
    stack = []
    stack.append((0, len(arr) - 1))
    while stack:
        low, high = stack.pop()
        if low < high:
            pivot = pivot_element(arr, low, high)
            stack.append((low, pivot - 1))
            stack.append((pivot + 1, high))
            
    return arr
    # Output Returning array should look like [['tconst','col1','col2'], ['tconst','col1','col2'], ['tconst','col1','col2'],.....]
    # column values in sublist must be according to the columns passed from the testcases.

#############################################################################################################
# Selection Sort
#############################################################################################################


def selection_sort(arr, columns):
    """
    arr: a list of lists representing the 2D array to be sorted
    columns: a list of integers representing the columns to sort the 2D array on
    Finally, returns the final sorted 2D array.
    """
    # NEED TO CODE
    # Implement Selection Sort Algorithm
    # return Sorted array
    arr_length = len(arr)
    columns_length = len(columns)
    col_type=type(arr[0][1])
    for x in range(arr_length-1):
        minIndex = x
        for y in range(x+1, arr_length):
            if col_type==str:
                if arr[y][1].strip() < arr[minIndex][1].strip():
                    minIndex = y
                elif arr[y][1].strip() == arr[minIndex][1].strip() and columns_length > 2:
                    for col_index in range(2, columns_length):
                        if type(arr[y][col_index])==str:
                            if arr[y][col_index].strip() < arr[minIndex][col_index].strip():
                                minIndex = y
                                break
                            elif arr[y][col_index].strip() > arr[minIndex][col_index].strip():
                                break
                        else:
                            if arr[y][col_index] < arr[minIndex][col_index]:
                                minIndex = y
                                break
                            elif arr[y][col_index] > arr[minIndex][col_index]:
                                break   
            else:
                if arr[y][1] < arr[minIndex][1]:
                    minIndex = y
                elif arr[y][1] == arr[minIndex][1] and columns_length > 2:
                    for col_index in range(2, columns_length):
                        if type(arr[y][col_index])==str:
                            if arr[y][col_index].strip() < arr[minIndex][col_index].strip():
                                minIndex = y
                                break
                            elif arr[y][col_index].strip() > arr[minIndex][col_index].strip():
                                break
                        else:
                            if arr[y][col_index] < arr[minIndex][col_index]:
                                minIndex = y
                                break
                            elif arr[y][col_index] > arr[minIndex][col_index]:
                                break

        arr[x], arr[minIndex] = arr[minIndex], arr[x]

    return arr
    # Output Returning array should look like [['tconst','col1','col2'], ['tconst','col1','col2'], ['tconst','col1','col2'],.....]
    # column values in sublist must be according to the columns passed from the testcases.

#############################################################################################################
# Heap Sort
#############################################################################################################


def max_heapify(arr, n, i, columns):
    """
    arr: the input array that represents the binary heap
    n: The number of elements in the array
    i: i is the index of the node to be processed
    columns: The columns to be used for comparison

    The max_heapify function is used to maintain the max heap property
    in a binary heap. It takes as input a binary heap stored in an array,
    and an index i in the array, and ensures that the subtree rooted at
    index i is a max heap.
    """
    left_node = 2*i+1
    right_node = 2*i+2
    largest_node = i
    col_length = len(columns)
    # if left_node < n and arr[left_node][1] >= arr[i][1]:
    #     largest_node = left_node
    if left_node < n and arr[left_node][1] >= arr[i][1]:
        if arr[left_node][1] > arr[i][1]:
            largest_node = left_node
        elif arr[left_node][1] == arr[i][1] and col_length > 2:
            for x in range(2, col_length):
                if arr[left_node][x] > arr[i][x]:
                    largest_node = left_node
                    break
                elif arr[left_node][x] < arr[i][x]:
                    break
    # else:
    #     largest_node = i
    # if right_node < n and arr[right_node][1] >= arr[largest_node][1]:
    #     largest_node = right_node
    if right_node < n and arr[right_node][1] >= arr[largest_node][1]:
        if arr[right_node][1] > arr[largest_node][1]:
            largest_node = right_node
        elif arr[right_node][1] == arr[largest_node][1] and col_length > 2:
            for x in range(2, col_length):
                if arr[right_node][x] > arr[largest_node][x]:
                    largest_node = right_node
                    break
                elif arr[right_node][x] < arr[largest_node][x]:
                    break

    if largest_node != i:
        arr[i], arr[largest_node] = arr[largest_node], arr[i]
        max_heapify(arr, n, largest_node, columns)


def build_max_heap(arr, n, i, columns):
    """
    arr: The input array to be transformed into a max heap
    n: The number of elements in the array
    i: The current index in the array being processed
    columns: The columns to be used for comparison

    The build_max_heap function is used to construct a max heap
    from an input array.
    """
    # NEED TO CODE
    # Implement heapify algorithm here
    for x in range((n)//2-1, -1, -1):
        max_heapify(arr, n, x, columns)


def heap_sort(arr, columns):
    """
    # arr: list of sublists which consists of records from the dataset in every sublists.
    # columns: store the column indices from the dataframe.
    Finally, returns the final sorted 2D array.
    """
    # NEED TO CODE
    # Implement Heap Sort Algorithm
    # return Sorted array
    n = len(arr)
    build_max_heap(arr, n, 0, columns)
    for x in range(len(arr)-1, 0, -1):
        arr[x], arr[0] = arr[0], arr[x]
        n = n-1
        max_heapify(arr, n, 0, columns)

    return arr
    # Output Returning array should look like [['tconst','col1','col2'], ['tconst','col1','col2'], ['tconst','col1','col2'],.....]
    # column values in sublist must be according to the columns passed from the testcases.

#############################################################################################################
# Shell Sort
#############################################################################################################


def shell_sort(arr, columns):
    """
    arr: a list of lists representing the 2D array to be sorted
    columns: a list of integers representing the columns to sort the 2D array on
    Finally, returns the final sorted 2D array.
    """
    # NEED TO CODE
    # Implement Shell Sort Algorithm
    # return Sorted array
    col_len=len(columns)
    n = len(arr)
    col_type=type(arr[0][1])
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i 
            if type(arr[j-gap][1])==str:
                arr[j-gap][1]=arr[j-gap][1].strip()
                temp[1]=temp[1].strip()
            while j >= gap and arr[j - gap][1] >= temp[1]:
                flag=0
                if col_type==str:
                    if arr[j-gap][1].strip()>temp[1].strip():
                        #print("entered ")
                        arr[j] = arr[j - gap]
                        j -= gap
                    elif arr[j-gap][1].strip()==temp[1].strip() and col_len>2:
                        #print("entered elif")
                        for x in range(2,col_len):
                            # print("inside ",i,j,x,col_len)
                            # print(arr[j-gap][x]<temp[x])
                            if type(arr[j-gap][x])==str:
                                if arr[j-gap][x].strip()>temp[x].strip():
                                    arr[j]=arr[j-gap]
                                    j-=gap
                                    break
                                elif arr[j-gap][x].strip()<temp[x].strip():
                                    # print("entered break break")
                                    flag=1
                                    break
                                elif arr[j-gap][x].strip()==temp[x].strip() and x==col_len-1:
                                    flag=1
                                    break
                            else:
                                if arr[j-gap][x]>temp[x]:
                                    arr[j]=arr[j-gap]
                                    j-=gap
                                    break
                                elif arr[j-gap][x]<temp[x]:
                                    # print("entered break break")
                                    flag=1
                                    break
                                elif arr[j-gap][x]==temp[x] and x==col_len-1:
                                    flag=1
                                    break
                    
                        if flag==1:
                            # print("entered fa")
                            break
                    else:
                        break
                else:
                    if arr[j-gap][1] > temp[1]:
                        #print("entered ")
                        arr[j] = arr[j - gap]
                        j -= gap
                    elif arr[j-gap][1]==temp[1] and col_len>2:
                        #print("entered elif")
                        for x in range(2,col_len):
                            # print("inside ",i,j,x,col_len)
                            # print(arr[j-gap][x]<temp[x])
                            if type(arr[j-gap][x])==str:
                                if arr[j-gap][x].strip()>temp[x].strip():
                                    arr[j]=arr[j-gap]
                                    j-=gap
                                    break
                                elif arr[j-gap][x].strip()<temp[x].strip():
                                    # print("entered break break")
                                    flag=1
                                    break
                                elif arr[j-gap][x].strip()==temp[x].strip() and x==col_len-1:
                                    flag=1
                                    break
                            else:
                                if arr[j-gap][x]>temp[x]:
                                    arr[j]=arr[j-gap]
                                    j-=gap
                                    break
                                elif arr[j-gap][x]<temp[x]:
                                    # print("entered break break")
                                    flag=1
                                    break
                                elif arr[j-gap][x]==temp[x] and x==col_len-1:
                                    flag=1
                                    break
                        if flag==1:
                            # print("entered fa")
                            break
                    else:
                        break
                
            arr[j] = temp
            #print("entered ",i,j)
        gap //= 2
    return arr
    # Output Returning array should look like [['tconst','col1','col2'], ['tconst','col1','col2'], ['tconst','col1','col2'],.....]
    # column values in sublist must be according to the columns passed from the testcases.

#############################################################################################################
# Merge Sort
#############################################################################################################


def merge(left, right, columns):
    """
    left: a list of lists representing the left sub-array to be merged
    right: a list of lists representing the right sub-array to be merged
    columns: a list of integers representing the columns to sort the 2D array on

    Finally, after one of the sub-arrays is fully merged, the function extends the result
    with the remaining elements of the other sub-array and returns the result as the final
    sorted 2D array.
    """
    # NEED TO CODE
    # Implement merge Logic
    # return Sorted array
    arr1 = []
    i, j = 0, 0
    col_len = len(left[0])
    m, n = len(left), len(right)
    while i < m and j < n:
        # print("entered the loop", i, j)
        if type(left[i][1])==str:
            #print("entered the str loop")
            if left[i][1].strip() <= right[j][1].strip():
                # print("entered if", i, j)
                arr1.append(left[i])
                i += 1
            else:
                # print("entered else", i, j)
                arr1.append(right[j])
                j += 1
        else:
            if left[i][1] <= right[j][1]:
                # print("entered if", i, j)
                if left[i][1] < right[j][1]:
                    # print("entered <", i, j)
                    arr1.append(left[i])
                    i += 1
                elif left[i][1] == right[j][1] and col_len <= 2:
                    # print("entered ==<", i, j)
                    arr1.append(left[i])
                    i += 1
                elif left[i][1] == right[j][1] and col_len > 2:
                    # print("entered == loop")
                    for x in range(2, col_len):
                        # print("entered the inner loop", i, j)
                        if left[i][x] < right[j][x]:
                            arr1.append(left[i])
                            i += 1
                            break
                        elif left[i][x] > right[j][x]:
                            arr1.append(right[j])
                            j += 1
                            break
                        elif left[i][x] == right[j][x] and x == col_len-1:
                            arr1.append(left[i])
                            i += 1
                            break
            else:
                # print("entered else", i, j)
                arr1.append(right[j])
                j += 1
    # print("exited the loop")
    for x in range(i, m):
        arr1.append(left[x])
    for y in range(j, n):
        arr1.append(right[y])
    return arr1


def merge_sort(data, columns):
    """
    data: a list of lists representing the 2D array to be sorted
    columns: a list of integers representing the columns to sort the 2D array on
    Finally, the function returns the result of the merge operation as the final sorted 2D array.
    """
    # NEED TO CODE
    if len(data) <= 1:
        return data
    mid = len(data)//2  # Mid value
    # Need to Code
    # Implement Merge Sort Algorithm
    # return Sorted array
    left = merge_sort(data[:mid], columns)
    right = merge_sort(data[mid:], columns)
    return merge(left, right, columns)
    # Output Returning array should look like [['tconst','col1','col2'], ['tconst','col1','col2'], ['tconst','col1','col2'],.....]
    # column values in sublist must be according to the columns passed from the testcases.

#############################################################################################################
# Insertion Sort
#############################################################################################################


def insertion_sort(arr, columns):
    """
    # arr: list of sublists which consists of records from the dataset in every sublists.
    # columns: store the column indices from the dataframe.
    Finally, returns the final sorted 2D array.
    """
    # NEED TO CODE
    # Insertion Sort Implementation
    # Return : List of tconst values which are obtained after sorting the dataset.
    arr_length = len(arr)
    columns_length = len(arr[0])
    for i in range(1, len(arr)):
        key = arr[i]
        for j in range(i-1, -1, -1):
            if arr[j][1] > key[1]:
                arr[j+1] = arr[j]
                arr[j] = key
            elif arr[j][1] == key[1] and columns_length > 2:
                flag = 0
                for x in range(2, columns_length):
                    if arr[j][x] > key[x]:
                        arr[j+1] = arr[j]
                        arr[j] = key
                    elif arr[j][x] < key[x]:
                        flag = 1
                        break
                if flag == 1:
                    break
            else:
                break

    return arr
    # Output Returning array should look like [['tconst','col1','col2'], ['tconst','col1','col2'], ['tconst','col1','col2'],.....]
    # column values in sublist must be according to the columns passed from the testcases.

#############################################################################################################
# Sorting Algorithms Function Calls
#############################################################################################################


def sorting_algorithms(file_path, columns, select):
    """
    # file_path: a string representing the path to the CSV file
    # columns: a list of strings representing the columns to sort the 2D array on
    # select: an integer representing the sorting algorithm to be used

    # colum_vals: is a list of integers representing the indices of the specified columns to be sorted.

    # data: is a 2D array of values representing the contents of the CSV file, with each row in
    the array corresponding to a row in the CSV file and each element in a row corresponding to a value
    in a specific column.

    More Detailed Description:

    df= #read imdb_dataset.csv data set using pandas library

    column_vals = #convert the columns strings passed from the test cases in the form of indices according to
                  #the imdb_dataset indices for example tconst column is in the index 0. Apart from the testcase
                  #Columns provided you must also include 0 column in the first place of list in column_vals
                  #for example if you have provided with columns {'startYear', 'primaryTitle'} which are in the
                  #indices {3,1}. So the column_vals should look like [0,3,1].

    data = #convert the dataframes into list of sublists, each sublist consists of values corresponds to
           #the particular columns which are passed from the test cases. In addition to these columns, each
           #sublist should consist of tconst values which are used to identify each column uniquely.
           #At the end of sorting all the rows in the dataset by using any algorithm you need to
           #Return : List of tconst strings which are obtained after sorting the dataset.
           #Example data looks like [['tconst string 1', 'startYear value 1', 'primaryTitle String 1'],
                                    #['tconst string 1', 'startYear value 1', 'primaryTitle String 1'],
                                    #................so on ]
                                    # NOTE : tconst string value must be in first position of every sublist and
                                    # the other provided column values with respect to columns from the provided
                                    # test cases must be after the tconst value in every sublist. Every sublist
                                    # Represents one record or row from the imdb_dataset.csv (sublist of values).
    """
    # NEED TO CODE
    # Read imdb_dataset.csv
    # write code here Inorder to read imdb_dataset
    # read imdb_dataset.csv data set using pandas library
    df = pd.read_csv(file_path)

    # convert the columns strings passed from the test cases in the form of indices according to
    column_vals = [
        0]+list(map(lambda column_val: column_names.index(column_val), columns))
    # the imdb_dataset indices for example tconst column is in the index 0. Apart from the testcase
    # Columns provided you must also include 0 column in the first place of list in column_vals
    # for example if you have provided with columns {'startYear', 'primaryTitle'} which are in the
    # indices {3,1}. So the column_vals should look like [0,3,1].

    # convert the dataframes into list of sublists, each sublist consists of values corresponds to
    data = df.iloc[:, column_vals].values.tolist()
    # the particular columns which are passed from the test cases. In addition to these columns, each
    # sublist should consist of tconst values which are used to identify each column uniquely.
    # At the end of sorting all the rows in the dataset by using any algorithm you need to
    # Return : List of tconst strings which are obtained after sorting the dataset.
    # Example data looks like [['tconst string 1', 'startYear value 1', 'primaryTitle String 1'],
    # ['tconst string 1', 'startYear value 1', 'primaryTitle String 1'],
    # ................so on ]
    # NOTE : tconst string value must be in first position of every sublist and
    # the other provided column values with respect to columns from the provided
    # test cases must be after the tconst value in every sublist. Every sublist
    # Represents one record or row from the imdb_dataset.csv (sublist of values).

#############################################################################################################
# Donot Modify Below Code
#############################################################################################################
    if (select == 1):
        start_time = time.time()
        output_list = insertion_sort(data, column_vals)
        end_time = time.time()
        time_in_seconds = end_time - start_time
        return [time_in_seconds, list(map(lambda x: x[0], output_list))]
    if (select == 2):
        start_time = time.time()
        output_list = selection_sort(data, column_vals)
        end_time = time.time()
        time_in_seconds = end_time - start_time
        return [time_in_seconds, list(map(lambda x: x[0], output_list))]
    if (select == 3):
        start_time = time.time()
        output_list = quicksort(data, column_vals)
        end_time = time.time()
        time_in_seconds = end_time - start_time
        return [time_in_seconds, list(map(lambda x: x[0], output_list))]
    if (select == 4):
        start_time = time.time()
        output_list = heap_sort(data, column_vals)
        end_time = time.time()
        time_in_seconds = end_time - start_time
        return [time_in_seconds, list(map(lambda x: x[0], output_list))]
    if (select == 5):
        start_time = time.time()
        output_list = shell_sort(data, column_vals)
        end_time = time.time()
        time_in_seconds = end_time - start_time
        return [time_in_seconds, list(map(lambda x: x[0], output_list))]
    if (select == 6):
        start_time = time.time()
        output_list = merge_sort(data, column_vals)
        end_time = time.time()
        time_in_seconds = end_time - start_time
        return [time_in_seconds, list(map(lambda x: x[0], output_list))]
