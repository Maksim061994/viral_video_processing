import os

def get_tags(data_sub, unwrapped_model):
    limit = 4000 if int(os.getenv("PARAM_N_CTX", "2048")) == 2048 else 6000
    query = f'''Ниже представлена текстовая расшифровка видео. Напиши подходящие для него хештеги, отдавай приоритет хештегам из одного слова

Расшифровка видео:
{data_sub['text'][:limit]}

Хештеги:
'''
    messages = [
        {'role': 'user',
         'content': query}
    ]

    outputs = unwrapped_model.predict(None, messages=messages)
    return outputs


def get_emojies(data_sub, unwrapped_model):
    limit = 4000 if int(os.getenv("PARAM_N_CTX", "2048")) == 2048 else 6000
    query = f'''Ниже представлена текстовая расшифровка видео. Напиши подходящие для него эмодзи не больше трёх

Расшифровка видео:
{data_sub['text'][:limit]}

Эмодзи:
'''
    messages = [
        {'role': 'user',
         'content': query}
    ]

    outputs = unwrapped_model.predict(None, messages=messages)
    return outputs


def get_summary(data_sub, unwrapped_model):
    limit = 4000 if int(os.getenv("PARAM_N_CTX", "2048")) == 2048 else 6000
    query = f'''Ниже представлена текстовая расшифровка видео. Напиши кратко о чём это видео

Расшифровка видео:
{data_sub['text'][:limit]}

'''
    messages = [
        {'role': 'user',
         'content': query}
    ]

    outputs = unwrapped_model.predict(None, messages=messages)
    return outputs


def get_llm_meta_info(data_sub, unwrapped_model):
    tags = get_tags(data_sub, unwrapped_model)
    emoji = get_emojies(data_sub, unwrapped_model)
    summary = get_summary(data_sub, unwrapped_model)

    return {
        'summary': summary,
        'tags': tags,
        'emoji': emoji
    }