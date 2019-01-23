# coding: utf-8
import json
import sys
import os
import asyncio
import re
import types
import enum
import cloudpickle as pickle
from enum import Enum
from copy import deepcopy
from jsonschema import Draft4Validator
from collections import ChainMap
from inspect import isgenerator, isgeneratorfunction
from typing import Pattern, Mapping, Iterable, AnyStr, Any, Dict, Set, Union, Hashable
from typing import List, Iterator, Generator, Callable, Optional
from operator import itemgetter, attrgetter, truth
from functools import partial, reduce
from itertools import chain, tee, repeat, starmap, groupby, zip_longest
from itertools import accumulate, permutations, product
from more_itertools import collapse, always_reversible
from more_itertools import always_iterable, flatten, ilen, first, seekable, spy
from datetime import datetime, date, time, tzinfo
from dateparser import parse as dateparse
from helpful_iterators.type_hints import OneNotIterableType


NOT_EMPTY_PATTER = re.compile('[\w|\d]+', flags=re.IGNORECASE)

# noinspection PyArgumentList
SEQUENCE_PRODUCER_TACTICS = Enum(
    'SEQUENCE_PRODUCER_TACTICS',
    [
        ('as_is', 'as_is'),
        ('as_part', 'as_part'),
        ('one_by_one', 'one_by_one'),
        ('cumulative', 'cumulative')
    ]
)




class CachedIterWrapper:
    """ Кэширующий итератор с возвожностью подавления ошибок """

    def __init__(self, iterable: Iterable, is_cached: bool = True, is_error_suppress: bool = True):
        iterable = always_iterable(iterable)
        self.is_cached = is_cached
        self.is_error_suppress = is_error_suppress
        self.iterable = seekable(iterable) if self.is_cached else iterable


    def __iter__(self) -> Iterator:
        return self._get_iter()


    def __next__(self) -> Any:
        ### Метод возможно нуждается в доработке! (проверить на отсутствие буферизации с опцией <is_cached: False>)

        try:
            # return next(self.iterable)  ### Старая версия
            return next(self._get_iter())

        except StopIteration:
            pass

        except Exception:
            if self.is_error_suppress: pass
            else: raise Exception('Iterator inner error')


    def _get_iter(self):
        """ Обработка внутренних ошибок итератора при вызовах метода iter() """

        if self.is_cached:
            self.iterable.seek(0)

        try:
            for x in self.iterable:
                yield x

        except Exception:
            # traceback.clear_frames(err.__traceback__)
            if self.is_error_suppress: pass
            else: raise Exception('Iterator inner error')




class FileOpeningHandler:
    """ Контекст-менеджер для обработки ошибок открытия файлов
        (недостаточно прав, файла не существует и т.д.)
    """
    def __init__(self, filename, error_message):
        self.filename = filename
        self.error_message = error_message

    def __enter__(self):
        return self.filename

    def __exit__(self, exc_type, exc_val, exc_tb):

        ### Если есть ошибка - выводим сообщение
        if exc_val:
            print(self.error_message)

        return True  ### отмечаем исключение как обработанное




class ExternalLamdaHandler:
    """ Контекст-менеджер для обработки ошибок в лямбда-функциях
        (для игнорирования ошибок в некорректных переданных лямба-функциях)
    """
    def __init__(self, lambda_func):
        self.lambda_func = lambda_func

    def __enter__(self):
        return self.lambda_func

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            pass  ### игнорируем ошибки лямбда-функций

        return True  ### отмечаем исключение как обработанное




class SuppressDefault:
    """ Менеджер контекста, способный возвращать значение при возникновении ошибки """

    def __init__(self, params: dict):
        self.params = params or {Exception: {'msg': 'Unexpected error', 'default': None}}
        self.default = None


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        config = self.params.get(exc_type, {})

        self.default = config.get('default')
        print(config.get('msg'), file=sys.stderr)

        return True




class CalculableIterFromGen:
    """ Вычисляемый(по желанию) итератор
        можно передавать в процессы/потоки - например в Pool.map()
    """

    def __init__(self, generator_func, *args, as_precalc=False, **kwargs):
        """
        :param param_func: Функция-генератор
        :param as_precalc: Вычислить значения сразу
        """
        self._generator_func = tuple(generator_func(*args, **kwargs)) if as_precalc else generator_func
        self._args = args
        self._kwargs = kwargs
        self._as_precalc = as_precalc


    def __iter__(self):
        ### Если выбран флаг <as_precalc> - вернуть вычисленные значения, иначе - вернуть генератор
        if self._as_precalc:
            return iter(self._generator_func)  ### функция iter() проверяет соответствие протоколу итератора
        else:
            return self._generator_func(*self._args, **self._kwargs)




def combine_params(
        schema_path: str,
        settings_path: Union[str, None] = None,
        without_none: bool = True,
        **params: Any

) -> Union[Mapping, None]:

    """ Скомбинировать параметры из json-файла и переданные в <params>

        :param schema_path: Путь к схеме файла json, описывающего параметры
        :param settings_path: Путь к файлу json, описывающего параметры
        :param without_none: Пропускать параметры с <None> значениями
    """

    if without_none:
        # noinspection PyTypeChecker
        params = dict(filter(itemgetter(1), params.items()))

    if settings_path:
        if os.path.isfile(settings_path):
            with open(schema_path, 'r') as schema_file:
                with open(settings_path, 'r') as settings_file:
                    settings_schema = json.load(schema_file)
                    settings_data = json.load(settings_file)

                    validator = Draft4Validator(settings_schema)
                    validator.validate(settings_data)

                    params = dict(ChainMap(settings_data, params))
        else:
            print('json settings file path received, but not exist', file=sys.stderr)

    return params



def maybe_pickled_unpicke(*args, **kwargs):
    """ Функция-декоратор для десериализации параметров """

    args = [
        pickle.loads(p) if isinstance(p, bytes) else p
        for p in args
    ]

    kwargs = {
        k: pickle.loads(v) if isinstance(v, bytes) else v
        for k, v in kwargs.items()
    }

    return args, kwargs




def get_files(
        directory: str,
        pattern: Pattern,
        incompleted_only=True,
        as_fullname=True,
        *completed_list: Mapping
) -> Iterable[str]:

    """ Поиск всех файлов по шаблону рекурсивно  от <directory> """

    files = (
        os.path.join(d, x) if as_fullname else (d, x)
        for d, dirs, files in os.walk(directory)
        for x in files if pattern.search(x)
    )

    result = filter(
        lambda f: f not in completed_list,
        files
    ) if incompleted_only else files

    return result




def date_autocomplete(t: time, d: date) -> datetime or time:
    """ Автодополнение, если дата неполная (только время)

        :param t: Время
        :param d: Дата
    """

    try:
        dt = datetime.combine(d, t)
        return dt

    except Exception:
        return t  # если ошибка - возвращаем переданное значение без изменений




def maybe_unicode_decode(value: AnyStr, encoding='utf-8') -> str:
    """ Если байт-строка - декодируем

        :param value: Строка или байт-строка
        :param encoding: Кодировка
    """

    result = value.decode(encoding) if isinstance(value, bytes) else value

    return result



def maybe_serialized_unpickle(obj: Any or bytes) -> Any:
    """ Если объект сериализован - восстанавливаем

        :param obj: Объект Python или сериализованный объект(байт-строка)
    """

    result = pickle.loads(obj) if isinstance(obj, bytes) else obj

    return result




def add_line_breaks(iterable: Iterable[AnyStr], encoding='utf-8', newline='\n') -> Iterable[Iterable[str]]:
    """ Добавляем переносы строки и декодируем (если это необходимо)

        :param iterable: Последовательность строк или байт-строк
        :param encoding: Кодировка
        :param newline: Символ переноса строки
    """

    result = map(
        lambda st: st+newline if st.rfind(newline) else st,

        map(
            partial(maybe_unicode_decode, encoding=encoding),
            iterable
        )
    )

    return result




def del_line_breaks(iterable: Iterable[str]) -> Iterable[str]:
    """ Удаляем переносы строк

        :param iterable: Последовательность строк
    """

    result = ( x.rstrip() for x in iterable )

    return result



def nested_getter(contain: Iterable[Any], *chain_indexes: Union[str, int], key_sep_pattern=r'[\s\.\,\[\]]', is_error_suppress=True, default_value=None):
    """ Рекурсивное получение вложенных элементов контейнера

        :param key_sep_pattern: Паттерн для разделения ключей, например вида 'a.b' или 'a,b'
        :param contain: Контейнер (последовательность переменной глубины вложенности)
        :param chain_indexes: Цепь индексов массива или словаря
        :param is_error_suppress: Режим подавления ошибок
        :param default_value: Значение возвращаемое при ошибке получения значения

        >>> dct = {'a1': {'a2': 1}, 'b1': {'b2': 2}}
        >>> print(nested_getter(dct, 'a1', 'a2'))
        1

        >>> lst = [[1, 2, 3], [4, 5, 6]]
        >>> print(nested_getter(dct, 1, 1))
        5
    """

    def _safe_getter(key, obj, is_error_suppress):
        buffer_result = obj
        parsed_keys = [ re.sub(r'[\s\'\"]+', '', str(x)) for x in re.split(key_sep_pattern, key) if x ] if isinstance(key, str) else repeat(key, 1)

        for k in parsed_keys:
            try:
                ### Сначала пытаемся получить значение, предполагая что объект - словарь
                buffer_result = itemgetter(k)(buffer_result)

            except (TypeError, KeyError):
                ### Если не получилось - пробуем получить значение через аттрибут объекта
                try:
                    buffer_result = attrgetter(k)(buffer_result)
                except Exception:
                    ### Иначе - действуем в соответствие с режимом подавления ошибок
                    if is_error_suppress:
                        buffer_result = default_value
                    else:
                        raise ValueError('value not found')

        return buffer_result


    result = reduce(
        lambda a, b: _safe_getter(key=b, obj=a, is_error_suppress=is_error_suppress),
        chain_indexes,
        contain
    )

    return result




def dict_slice(
    *dicts: dict,
    keys: Iterable[Union[int, str]],
    is_safe_getter: bool=True,
    default_value: bool=None,

) -> Iterable[dict]:

    """ Срез словаря по ключам

        :param keys: Ключи для среза
        :param is_safe_getter: Безопасное получение элементов (при отсутствии - замена на <default_value>)
        :param default_value: Значение по-умолчанию
        :param d: Словарь\словари
    """

    if is_safe_getter:
        results = map((lambda d: {k: d.get(k, default_value) for k in keys}), dicts)
    else:
        results = map(itemgetter(*keys), dicts)

    return results




# noinspection PyUnusedLocal
def flatly_repeat(value: Iterable or OneNotIterableType, n: int=1, *args, **kwargs) -> Iterable:
    """ Повторяет переданный элемент n-раз
        Если елемент - массив значений, то функция разворачивает его

        >>> print(list(flatly_repeat('a', n=3)))
        ['a', 'a', 'a']

        >>> print(list(flatly_repeat(['a', 'b'], n=3)))
        ['a', 'b', 'a', 'b', 'a', 'b']
    """

    results = flatten( repeat(tuple(always_iterable(value)), n) )

    return results




def separate_dict_by_values(
    dct: Dict[Hashable, Iterable],
    is_empty_prevent: bool = True,
    default: Any = None

) -> Iterable[dict]:

    """ Создание последовательности словарей по 1 строке(шапке) и остальным строкам
        (Декартово произведение множеств-значений словаря)

        :param dct: Словарь со значениями-последовательностями
        :param is_empty_prevent: Заполнять пустые массивы <None> значениями
        :param default: Значение для заполнения пустых массивов в режиме <is_empty_prevent>
    """

    if dct:
        headers, values_pack = zip(*dct.items())
        if is_empty_prevent:
            values_pack = ( x or [default] for x in values_pack )

        results = [ dict(zip(headers, v)) for v in product(*values_pack) ]
        return results

    else:
        return []




def get_futures_results(futures_results: Iterable) -> Dict[str, Set]:
    """ Результаты функции <wait> concurrent futures

        :param futures_results: Последовательность асинхронных задач
    """

    result = dict(
        zip(
            ('success', 'errors'),
            map(
                lambda res: set(x.result() for x in res),
                futures_results
            )
        )
    )

    return result




def get_len_safely(value: Any, default: int=1, return_copy: bool=False):
    # noinspection PyUnresolvedReferences
    """ Количество элементов в последовательности
        (Возможно передать единственное значение, тогда очевидно длина = 1)

        :param value: Единичное значение или последовательность
        :param default: Значение, возвращаемое если последовательность пустая, либо передано <None>
        :param return_copy: Вернуть копию, в случае если был передан генератор

        >>> gen_len, gen_copy = get_len_safely((x for x in range(3)), return_copy=True)
        >>> print("Generator len: {}".format(gen_len), '|', "Generator copy: {}".format(list(gen_copy)))
        Generator len: 3 | Generator copy: [0, 1, 2]

        >>> lst_len = get_len_safely(list(range(3)))
        >>> print("Iterable len: {}".format(lst_len))
        Iterable len: 3

        >>> one_value_len = get_len_safely(3)
        >>> print("One value len: {}".format(one_value_len))
        One value len: 1
    """

    iterable, iter_copy = tee(always_iterable(value)) if (is_lazy_sequence(value) and return_copy) else [always_iterable(value), None]
    iter_len = ilen(iterable) or default

    result = (iter_len, iter_copy) if iter_copy else iter_len

    return result




def get_first_safely(value, default_value=None) -> Any:
    """ Безопасно(без ошибки выхода за пределы массива) получить первый элемент
        Если <value> НЕитерируемое значение - возвращаем его
    """

    result = first(always_iterable(value), default=default_value)

    return result



def itemgetter_safe(iterabe: Iterable, *keys: Union[Any, Iterable[Any]]) -> List[Any]:
    """ ВСЕГДА возвращает итерируемую последовательность функции <operator.itemgetter>
        (обычный <itemgetter> может возвращать единичное значение в случае передачи ему одного ключа)
    """

    results = itemgetter(*keys)(iterabe)

    if len(keys) == 1:
        results = [results, ]

    return results



def sorted_alnum(iterable: Iterable[Any]) -> List[Any]:
    """ Сортировка элементов последовательности

        >>> lst = ['30', '10', '03', '02', 1, 'ЭС-01', 'ЭС-10']
        >>> print(sorted_alnum(lst))
        [1, '02', '03', '10', '30', 'ЭС-01', 'ЭС-10']

        >>> dct = {'30': [1], '10': [2], '03': [3], '02': [4], 1: [5], 'ЭС-01': [6], 'ЭС-10': [7]}
        >>> sorted_dct = dict(zip(sorted_alnum(dct.keys()), dct.values()))
        >>> print(sorted_dct)
        {1: [1], '02': [2], '03': [3], '10': [4], '30': [5], 'ЭС-01': [6], 'ЭС-10': [7]}
    """

    # noinspection PyTypeChecker
    def _sorted_func(key):
        return [
            get_number_represent(k, as_int=True) or str(k)
            for k in re.split(r'(\d+)', str(key))
        ]

    results = sorted(always_iterable(iterable), key=_sorted_func)
    return results



def sorted_dict_by_keys(d: dict, drop_none_values: bool = False) -> dict:
    """ Сортировка ключей результирующего словаря """

    filter_func = truth if drop_none_values else None

    sorted_keys = sorted_alnum(d.keys())
    sorted_values = itemgetter_safe(d, *sorted_keys)

    result = dict(zip(
        sorted_keys,
        [list(filter(filter_func, x)) for x in sorted_values]
    ))

    return result



def sorted_with_none(iterable: Iterable, none_first: bool=True) -> List:
    """ Сортировка последовательности с <None> элементами

        :param iterable: Последовательность для сортировки
        :param none_first: Вывести элементы <None> первыми (иначе - последними)
    """

    results = sorted(
        iterable,
        key=lambda x: (x is not None, x) if none_first else (x is None, x)
    )

    return results




def get_number_represent(x: Any, default: Any = None, as_int: bool = False) -> Union[float, None]:
    """ Безопасное получение числа (например из строки содержащей число) """

    if not (isinstance(x, int) or isinstance(x, float)):
        try:
            x = int(x) if as_int else x
        except (TypeError, ValueError):
            x = default

    return x



def get_string_represent(x: Any, default: Any = None) -> Union[str, None]:
    """ Безопасное получение строкового представления (например из даты) """

    if isinstance(x, (datetime, date, time)):
        x = x.isoformat()
    elif isinstance(x, enum.Enum):
        x = str(x.value)
    else:
        try:
            x = str(x)
        except (TypeError, ValueError):
            x = default

    return x




def get_date_strict(
        d: Union[datetime, date, str],
        date_locales: Union[Iterable[str], str] = ('ru', 'en'),
        tzinfo: Union[tzinfo, None] = None,
        as_timestamp: bool = False

) -> Union[int, float, datetime, date]:

    """ Получить дату из строки, либо вернуть исходную (если это уже объект даты Python) """

    date_locales = list(always_iterable(date_locales))

    result = d if isinstance(d, (datetime, date)) else dateparse(d, languages=date_locales, settings={'STRICT_PARSING': True})

    if isinstance(result, datetime):
        result = result.replace(tzinfo=tzinfo)  ### аргумент <tzinfo> есть только у объектов <datetime>
        result = result.timestamp() if as_timestamp else result  ### <timestamp> есть только у объектов <datetime>

    return result




def get_dateperiod(
    start_date: Union[datetime, date, str],
    end_date: Union[datetime, date, str],
    start_default: Union[datetime, date],
    end_default: Union[datetime, date],
    as_imap_dateperiod: bool = False,
    languages: Union[Iterable[str], str] = ('ru', 'en')

) -> Union[Union[datetime, date], str]:

    """ Получить начальную и конечную дату из 'сырых' строк """

    sd, ed = sorted_with_none(
        map(
            partial(get_date_strict, date_locales=languages, as_timestamp=False),
            [start_date, end_date]
        )
    )

    ### Нормализованный интервал дат (обработка неполных дат)
    sd_norm, ed_norm = starmap(
        lambda a, b: a or b,
        zip([sd, ed], sorted([start_default, end_default]))
    )

    ### Если требуется возвращаем даты в виде шаблона IMAP
    if as_imap_dateperiod:
        imap_date_pattern = '%d-%b-%Y'
        strf_sd, strf_ed = sd_norm.strftime(imap_date_pattern), ed_norm.strftime(imap_date_pattern)

        if ed_norm.date() >= datetime.now().date():
            results = "(since {0})".format(strf_sd)  # Возможно IMAP стандарт не умеет обрабатывать даты превосходящие текущую
        else:
            results = "(since {0} before {1})".format(strf_sd, strf_ed)

    else:
        results = (sd_norm, ed_norm)

    return results




def apply(func, *args, **kwargs):
    """ Вызов функции в функциональном стиле """

    result = func(*args, **kwargs)

    return result



def create_aio_rootpoint(coro: asyncio.coroutine, *args, **kwargs):
    """ Обертка для передачи корутин в параллельные процессы/потоки
        (фактически создается новый подцикл событий)
    """

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete( coro(*args, **kwargs) )

    loop.close()

    return result





def groupby_records_iterable(
        records_pack_iterable,
        sort_keys,
        group_keys,
        getter_func=itemgetter
):
    """ группировка по ключам """

    ## Старая версия (стабильная но не работает сортировка с None)
    sorted_list = sorted(
        records_pack_iterable,
        key=getter_func(*sort_keys)
    )

    # ### Новая НЕтестированная версия (устойчива к наличию None в полях записи)
    # sorted_list = sorted(
    #     records_pack_iterable,
    #     key=lambda x: list(map(bool, always_iterable(getter_func(*sort_keys)(x))))
    # )

    results = (
        tuple(always_reversible(group_values))
        for group_name, group_values in groupby(sorted_list, key=getter_func(*group_keys))

    )  # например: [[1,2,3], [4,5,6]]

    results = zip_longest(*results, fillvalue=[])  # например: [(3, 6), (2, 5), (1, 4)]

    return next(results, []), collapse(results)




def dict_update_nested(dct, keys_path, new_val):
    """ Обновление словаря по вложенным ключам (пути к значению)
        Если такого пути не существует - возвращается копия исходного словаря

        :param dct: Исходный словарь
        :param keys_path: Итерационный путь (массив ключей) для нахождения значения для замены
        :param new_val: Значение для замены
    """

    result = deepcopy(dct)  ### Полностью копируем словарь в новый объект (не изменяем исходный)

    ### Несмотря на то что reduce - это элемент функционального программирования,
    ### в ситуации со словарем мы оперируем ссылками, поэтому по ним можем изменять значения внешнего объекта
    val_link = reduce(
        lambda key, d: dict.get(key, d, {}),
        keys_path,
        result
    )

    val_link.update(new_val)

    return result



def drop_duplicate_records(records: Iterable[Any]) -> List[Any]:
    """ Удаление повторяющихся записей в массиве записей

        >>> rec_list = [{'id': 2, 'points': [1,2,3]}, {'id': 2, 'points': [1,2,3]}, {'id': 1, 'points': [1,2,3]}]
        >>> print(drop_duplicate_records(rec_list))
        [{'id': 2, 'points': [1, 2, 3]}, {'id': 1, 'points': [1, 2, 3]}]
    """

    results = []
    seen = []
    for rec in records:
        if rec not in seen:
            seen.append(rec)
            results.append(rec)

    return results



def is_lazy_sequence(iterable: Iterable) -> bool:
    """ Является ли данная последовательность генератором или итератором (в смысле "ленивой последовательности")
        Работает для builtin-генераторов(map, zip и т.п.), определенных в функциях через <yield> или вида (x for x in range(10))

        >>> gen = range(3)
        >>> print(is_lazy_sequence(gen))
        True

        >>> lst = list(range(3))
        >>> print(is_lazy_sequence(lst))
        False
    """

    result = any(
        f(iterable) for f in [
            (lambda x: isinstance(x, Iterator)),
            isgeneratorfunction,

            # дополнительные проверки ...
            isgenerator,
            (lambda x: isinstance(x, Generator))
        ]
    )

    return result





def _cumulative_sequence(iterable: Iterable, is_cached: bool=True, is_error_suppress: bool=True) -> CachedIterWrapper:
    """ Кумулятивное накопление массива из элементов последовательности

        >>> print(list(_cumulative_sequence(range(3))))
        [(0,), (0, 1), (0, 1, 2)]

        >>> print(list(_cumulative_sequence(['a', 'b', 'c'])))
        [('a',), ('a', 'b'), ('a', 'b', 'c')]
    """

    try:
        iterable = always_iterable(iterable)
        head = next(iterable)

        results = accumulate(
            chain(repeat(tuple([head]), 1), iterable),
            lambda acc, x: (*acc, x)
        )

    except StopIteration:
        results = []

    return CachedIterWrapper(results, is_cached, is_error_suppress)





def _chain_from_iterable_beside(iterable: Iterable, ignored_type: type=str, is_cached: bool=True, is_error_suppress: bool=True) -> CachedIterWrapper:
    """ Раскрывает последовательности вложенные в <iterable> в единую последательность,
            если только они не являются экземплярами <ignored_type> (по-умолчанию строки)
        Важно: Внутренние последовательности должны быть однородными (вывод о содержании <iterable> делается по первому элементу)

        >>> print('Sequence of strings: ', list(_chain_from_iterable_beside(['first', 'second'])))
        Sequence of strings:  ['first', 'second']

        >>> print('Sequence of generators: ', list(_chain_from_iterable_beside([range(3), range(5)])))
        Sequence of generators:  [0, 1, 2, 0, 1, 2, 3, 4]

        >>> print('Sequence of numbers: ', list(_chain_from_iterable_beside([1, 2, 3])))
        Sequence of numbers:  [1, 2, 3]
    """

    head, results = spy(always_iterable(iterable))  # можно передавать не только последовательность, но и одно значение

    if head and isinstance(head[0], Iterable) and (not ignored_type or not isinstance(head[0], ignored_type)):
        results = chain.from_iterable(iterable)

    return CachedIterWrapper(results, is_cached, is_error_suppress)





def sequence_producer(
    iterable: Any,
    strategy: SEQUENCE_PRODUCER_TACTICS=SEQUENCE_PRODUCER_TACTICS.as_is,
    ignored_type: Optional[type]=str,
    is_cached: bool=True,
    is_error_suppress: bool=True
) -> Iterable:

    """ Выдача элементов последовательности в соответствии с определенной стратегией

        >>> print(list(sequence_producer(['one', 'two'], strategy=SEQUENCE_PRODUCER_TACTICS.as_is)))
        ['one', 'two']

        >>> print(list(sequence_producer(['one', 'two'], strategy=SEQUENCE_PRODUCER_TACTICS.as_part)))
        [['one', 'two']]

        >>> print(list(sequence_producer(['one', 'two'], strategy=SEQUENCE_PRODUCER_TACTICS.one_by_one, ignored_type=None)))
        ['o', 'n', 'e', 't', 'w', 'o']

        >>> print(list(sequence_producer(['one', 'two'], strategy=SEQUENCE_PRODUCER_TACTICS.cumulative)))
        [('one',), ('one', 'two')]
    """

    mapping = {
        SEQUENCE_PRODUCER_TACTICS.as_is: always_iterable(iterable),
        SEQUENCE_PRODUCER_TACTICS.as_part: repeat(iterable, 1),
        SEQUENCE_PRODUCER_TACTICS.one_by_one: _chain_from_iterable_beside(iterable, ignored_type, is_cached, is_error_suppress),
        SEQUENCE_PRODUCER_TACTICS.cumulative: _cumulative_sequence(_chain_from_iterable_beside(iterable, ignored_type, is_cached, is_error_suppress))
    }

    results = mapping.get(strategy, [])

    return results





def filter_map_reduce(
    sources: Iterable,
    grouper: Union[Callable, None] = None,
    filters: Union[Iterable[Callable], None] = None,
    modifiers: Union[Iterable[Callable], None] = None,
    reducers: Union[Iterable[Callable], None] = None,
    context: Union[Any, None] = None,
    filter_value_getter: Union[Callable, None] = None,
    filter_result_getter: Union[Callable, None] = None,
    is_cached: bool = True,
    is_error_suppress: bool = True

) -> CachedIterWrapper:

    # noinspection PyUnresolvedReferences
    """ Функция-интерфейс для реализации настраиваемых действий в парадигме <filter-map-reduce>

            >>> import re
            >>> func = lambda x, y: x+y

            >>> FILTERS = [(lambda x, ctx=None: re.search(r'[\w]', x, flags=re.IGNORECASE)), ]
            >>> MODIFIERS = [(lambda x, ctx=None: x.casefold()), (lambda x, ctx=None: x.split()), ]
            >>> REDUCERS = [(lambda iterable, ctx=None: next(iterable, [])), ]

            >>> tokens = filter_map_reduce(['Test File One', ], filters=FILTERS, modifiers=MODIFIERS, reducers=REDUCERS)
            >>> print('Found tokens :', list(tokens))
            Found tokens : ['test', 'file', 'one']
        """

    context = context and CachedIterWrapper(context)  # кэшируем контекст

    filter_value_getter = filter_result_getter or (lambda x: x[0] if isinstance(x[0], Iterable) else x[1])
    filter_result_getter = filter_result_getter or (lambda x: x[1])
    filters = filters and [ partial(func, ctx=context) for func in filters ]
    modifiers = modifiers and [ partial(func, ctx=context) for func in modifiers ]
    reducers = reducers and [ partial(func, ctx=context) for func in reducers ]


    ### Группируем результаты, если есть соответствующая функция, иначе - пронумерованная последовательность
    results = (
        (key, tuple(group))
        for key, group in groupby(sources, key=grouper)

    ) if grouper else ((i, x) for i, x in enumerate(sources))


    ### Если предоставлена функция группировки, то фильтрация идет по ключам группировки
    results = map(
        filter_result_getter,
        filter(
            lambda x: all(catch(func, filter_value_getter(x), default=True) for func in filters),
            results

        ) if filters else results
    )


    ### К каждому элементу поочередно применяем функции-модификаторы
    results = map(
        lambda s: reduce(lambda x, func: catch(func, x), modifiers, s),
        results

    ) if modifiers else results


    ### Для вывода конечного результата используем функции-редуценты
    ### обертка <reduce> в <always_iterable> нужна для вывода единообразного типа результатов (последовательности)
    results = reduce(
        lambda x, func: catch(func, x),
        reducers,
        results

    ) if reducers else results


    ### Несмотря на то, что мы пытаемся разрешить ошибки вызванные переданными внешними <lambda> при помощи <catch>, ...
    ### ... они могут возникнуть внутри самой логики <built-in> генераторов (groupby, map, reduce)
    ### Если включено подавление ошибок, то при их возникновение возвращаем пустой итератор вместо возбуждения исключения
    results = CachedIterWrapper(results, is_cached, is_error_suppress)

    return results




def catch(func: Callable, *args, default: Any=None, error_class: Exception=Exception, verbose_mode=False, **kwargs) -> Any:
    # noinspection PyTypeChecker
    # noinspection PyUnresolvedReferences
    """ Перехват ошибок выполнения функции (для контроля ошибок функций внутри итераторов)

        >>> print(catch((lambda x, y: x/0+ y), 3, y=5, error_class=ZeroDivisionError, error_message='Cannot be divided by zero!'))
        Cannot be divided by zero!
        None
    """

    try:
        return func(*args, **kwargs)

    except error_class as err:
        if verbose_mode:
            print(err)
        return default




def reform_date_by_mask(dt: date, dt_mask: date) -> date:
    """ Перестановка элементов даты в соответствие с маской (другой датой)

        >>> from datetime import date

        >>> date_mask = date(2018, 12, 22)
        >>> doubtful_date = date(2022, 11, 18)

        >>> print(reform_date_by_mask(doubtful_date, date_mask))
        2018-11-22
    """

    def _check_date(date_tuple: tuple) -> bool:
        try:
            date(*map(int, date_tuple))
            is_valid_date = True
        except ValueError:
            is_valid_date = False
        return is_valid_date


    result = min(
        map(
            lambda x: datetime.strptime('-'.join(x), '%y-%m-%d').date(),
            filter(
                _check_date,
                permutations(dt.strftime('%y-%m-%d').split('-'))
            )
        ),
        key=lambda x: abs(x - dt_mask)  # условие - минимальная разница между датой и ориентировочной датой
    )

    return result




def dataclass_pickle_prepair(*classes) -> None:
    """ Подготовка некоторых классов для сериализации (namedtuple, dataclass)

        :type classes: объекты классов
    """

    for cls in classes:
        setattr(types, cls.__name__, cls)
