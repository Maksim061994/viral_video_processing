from func_test import func_test


def main(path_to_vid):
    data_api = func_test(path_to_vid)

    return data_api


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_api = main('data/song_laught.mp4')
