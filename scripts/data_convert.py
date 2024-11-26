import csv
import os

import pandas as pd
from sklearn.model_selection import train_test_split

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

    if not check_category(category):
        raise ValueError(f'Invalid category: {category}')

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

    # result_df를 category 칼럼에 대해 train/test stratified sampling
    result_df = pd.DataFrame(results)
    print(f'count: {len(result_df)}')

    # 카테고리별로 갯수 확인
    print("\nCategory Count:")
    print(result_df['category'].value_counts())

    # category별로 train과 test로 나누는 함수
    def split_data_by_category(df, test_size=0.2, random_state=42):
        train_list = []
        test_list = []

        # category별로 데이터 나누기
        for category, group in df.groupby('category'):
            train, test = train_test_split(
                group, test_size=test_size, random_state=random_state
            )
            train_list.append(train)
            test_list.append(test)

        # 결과를 다시 DataFrame으로 결합
        train_df = pd.concat(train_list).reset_index(drop=True)
        test_df = pd.concat(test_list).reset_index(drop=True)

        return train_df, test_df

    # 데이터 분리 실행
    train_df, test_df = split_data_by_category(result_df)

    print("\nTest Data:")
    print(test_df)

    # save output to csv
    def save_csv(df, file_path):
        df.to_csv(file_path, index=False)

    save_csv(train_df, 'train.csv')
    save_csv(test_df, 'test.csv')


if __name__ == '__main__':
    main()
