import csv
import os

from common import APP_ROOT


# read csv file
def read_csv(file_path: str, encoding: str = 'utf-8') -> list:
    with open(file_path, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        data = []

        for row in reader:
            data.append(row)

    return data


def check_category(category):
    defined_categories = ['진로탐색', '자아탐색', '고교학점제', '학업', '입시']

    # remove space in category
    category = category.replace(' ', '')

    if category not in defined_categories:
        return False

    return True


def process_data(rows):
    dialog = []

    for row in rows:
        speaker = row[2]
        content = row[3]

        dialog.append(f'{speaker} : {content}')

    id = rows[0][0]
    category = rows[0][1]

    if len(category) == 0:
        raise ValueError(f'Invalid category: {category}')

    # if not check_category(category):
    #     raise ValueError(f'Invalid category: {category}')

    data_dict = dict()
    data_dict['id'] = id
    data_dict['category'] = category
    data_dict['content'] = '\n'.join(dialog)

    return data_dict


def main():
    csv_file = os.path.join(APP_ROOT, 'data/(사자가온다) 초기100건 데이터수집.csv')

    data = read_csv(csv_file, encoding='utf-8-sig')

    start_row = 0
    results = []
    for i, row in enumerate(data):
        if i == 0:
            continue

        col0 = row[0]
        col1 = row[1]
        col2 = row[2]
        col3 = row[3]

        if col0 != '':
            if start_row == 0:
                start_row = i
            else:
                end_row = i
                print(f'{start_row} ~ {end_row}')

                result = process_data(data[start_row:end_row])
                results.append(result)
                print(result)
                # break

                start_row = i
        elif col2 == '':
            end_row = i - 1
            print(f'{start_row} ~ {end_row}')

            result = process_data(data[start_row:end_row])
            results.append(result)
            print(result)
            break  # 종료조건

        print(row)

    # save output to csv
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        # write header
        writer.writerow(['id', 'category', 'content'])

        for result in results:
            writer.writerow([result['id'], result['category'], result['content']])

    print(f'Output saved to output.csv')

if __name__ == '__main__':
    main()
