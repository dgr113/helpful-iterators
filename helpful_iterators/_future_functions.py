# coding: utf-8

""" Functions for future use only """

from copy import deepcopy
from functools import reduce, partial
from itertools import chain, tee, compress, cycle
from typing import Iterable




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



def reshape_seq_strings(iterable, is_casefold_out: bool=True) -> Iterable[str]:
    """ Перевод последовательно предложений в массив отдельных слов """

    results = chain.from_iterable(
        str(x).split()
        for x in chain.from_iterable(iterable)
    )

    return map(lambda x: x.casefold(), results) if is_casefold_out else results



def get_continuous_sequence(iterable):
    """ Получить последовательность с парными значениями между оригинальными """

    a, b, c = tee(iterable, 3)

    points = compress(a, cycle([1, 0]))
    gaps = zip(b, next(c, None))

    results = chain.from_iterable( zip(points, gaps) )

    return results



def partial_multiple(functions, *args, **kwargs):
    """ Частичное дополнение аргументов функций <functions> из переданных переданных именованных и неименованных аргументов """

    results = [ partial(func, *args, **kwargs) for func in functions ]

    return results



def apply(func, *args, **kwargs):
    """ Вызов функции в функциональном стиле """

    result = func(*args, **kwargs)

    return result
