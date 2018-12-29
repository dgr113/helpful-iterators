# coding: utf-8

from typing import Tuple, Iterable, Any, Dict, Hashable, Callable, Generator, Set, List, Mapping


DataRowType = Iterable[Any]
DataColumnType = Iterable[Any]
DataRowsPack = Iterable[DataRowType]

OneNotIterableType = Any  ### any NOT ITERABLE value
ActionMappingType = Dict[Hashable, Callable]
MergedCellsType = Iterable[Tuple[int, int, int, int]]
MergedValuesType = Tuple[Tuple, OneNotIterableType, Any]
XlDataType = Generator[DataRowType, None, None]
XlDataPackType = Tuple[XlDataType, MergedValuesType]

# AggregatorType = Callable[[Any, Any], Any]
RecImageType = Dict[str, Any]
AggregatorType = Callable[[RecImageType, Any, Any], Any]
SeparatorType = Callable[[Any], Iterable]
HeaderType = Tuple[str, Set[str], AggregatorType, SeparatorType]
ColumnNamesDictType = Dict[str, Tuple[Tuple[int, ...], int, AggregatorType, SeparatorType]]
ColumnNamesListType = Tuple[Tuple[str, List[int], int, AggregatorType, SeparatorType], ...]

ClassificatorType = Tuple[Any, str, List[int]]
ClassifyFuncType = Callable[[Any], bool]
ClassificatorTypeMap = Mapping[int, ClassificatorType]

FilterFuncType = Callable[..., bool]
ModifierFuncType = Callable[..., Any]

RawRecImageType = Dict[str, Iterable[Any]]
RecPackImageType = Tuple[RecImageType, ...]

DatetupleType = Tuple[int, int, int]
