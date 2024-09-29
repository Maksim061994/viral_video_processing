import datetime
import os
import json

def get_timedelta(str_time):
    t = datetime.datetime.strptime(str_time, "%H:%M:%S")
    delta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    return delta


# эту нужно обернуть в трай
def get_llm_interval(data_sub, unwrapped_model):
    result_text = ''
    limit = 4000 if int(os.getenv("PARAM_N_CTX", "2048")) == 2048 else 6000

    for segment in data_sub['segments']:
        start_sec = datetime.timedelta(seconds=int(segment['start']))
        end_sec = datetime.timedelta(seconds=int(segment['end']))
        seg_text = segment['text']
        seg_text = f'''{{'start': {start_sec}, 'end': {end_sec}, 'text': {seg_text}}}, \n'''
        result_text += seg_text
        if len(result_text) > limit:
            break

    query = f'''Ниже представлена текстовая расшифровка видео с таймкодами. Найди самые интересные по смыслу места, они должны идти последовательно.
    В списке должно быть не больше десяти элементов.
    
    Дай ответ ввиде списка json
    {{
        "start": "hh:mm:ss", 
        "end": "hh:mm:ss", 
        "text": "str"
    }}
    
    Расшифровка видео:
    {result_text}
                
    Отрезки:
    '''

    messages = [
        {
            'role': 'user',
            'content': query
        }
    ]

    # TODO доделать
    outputs = unwrapped_model.predict(None, messages=messages)

    parsed_ans = json.loads(outputs.split('```json')[1][:-3])

    duration = get_timedelta(parsed_ans[-1]['end']).seconds - get_timedelta(parsed_ans[0]['start']).seconds
    if duration > 180:
        end = get_timedelta(parsed_ans[0]['start']) + datetime.timedelta(seconds=180)
    else:
        end = get_timedelta(parsed_ans[-1]['end'])
    final_interval = [get_timedelta(parsed_ans[0]['start']), end, 'LLM']

    return final_interval


def is_overlap(interval_1, interval_2):
    min_len = min([interval_1[1].seconds - interval_1[0].seconds, interval_2[1].seconds - interval_2[0].seconds])

    latest_start = max([interval_1[0].seconds, interval_2[0].seconds])
    earliest_end = min([interval_1[1].seconds, interval_2[1].seconds])

    overlap = max([0, earliest_end - latest_start])

    return overlap > (min_len / 2)


def insert_llm_interval(final_interval, intervals):
    combined_intervals = []

    for interval in intervals:
        if not is_overlap(final_interval, interval):
            combined_intervals.append(interval)
        else:
            final_interval[2] = 'LLM + ML'

    combined_intervals.append(final_interval)

    return combined_intervals