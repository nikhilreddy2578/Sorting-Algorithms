import csv
import os
import heapq


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1


def write_to_file(file_path, data):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def read_from_file(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
        return data


def data_chuncks(file_path, columns, memory_limitation):
    # Create directory for individual files
    if not os.path.exists('./Individual'):
        os.makedirs('./Individual')

    # Read in data from file in chunks and sort each chunk using merge sort
    chunk_num = 1
    while True:
        # Read in next chunk of data
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= (chunk_num-1) * memory_limitation and i < chunk_num * memory_limitation:
                    data.append(row)
        if not data:
            break

        # Sort the data using merge sort
        merge_sort(data)

        # Write the sorted chunk to a new file
        output_file = f'./Individual/sorted_{chunk_num}.csv'
        write_to_file(output_file, data)

        chunk_num += 1


def Mystery_Function(file_path, memory_limitation, columns):
    # Create directory for final files
    if not os.path.exists('./Final'):
        os.makedirs('./Final')

    # Load sorted chunks into memory
    sorted_chunks = []
    for i in range(1, 94):
        file_path = f'./Individual/sorted_{i}.csv'
        data = read_from_file(file_path)
        sorted_chunks.append(data)

    # Use heapq.merge to merge the sorted chunks
    # 2000 records are loaded from each chunk at a time
    chunk_idx = 0
    while True:
        chunk = sorted_chunks[chunk_idx]
        if not chunk:
            break

        # Load 2000 records from current chunk
        records = []
        while len(records) < 2000 and chunk:
            records.append(chunk.pop(0))

        # Check if we've reached the end of the current chunk
        if not chunk:
            chunk_idx += 1

        # Merge the records with the existing sorted records using heapq.merge
        merged_records = []
        for record in heapq.merge(records, sorted_chunks):
            merged_records.append(record)

            # Write out the merged records when we
