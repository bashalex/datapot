* Полагаю, что хранить данные внутри библиотеки удобнее всего на основе pandas.DataFrame.
Это, кажется, общепринятый формат, который при необходимости конвертируется в любые другие, и с ним удобно работать.

*Все названия очень условные.*
*Множество моментов здесь не упомянуто, часть не до конца продумана.*

### Proposal
* Для начала у нас должен быть модуль, который приняв на вход json будет возвращать полученный из него DataFrame.
Скорее всего на этом же этапе он должен и размечать какие колонки к какому типу относятся - текст, категории, время и т.д.
При этом функция разметки, разумеется, должна быть вынесена отдельно, чтобы была возможность применять ее и просто к DataFrame'у. Назовем этот модуль JsonConverter:
``` 
- datapot
    __init__.py
    - JsonConverter
        __init__.py
```

* Для того, чтобы хранить где-то нашу разметку создадим клаcс Pot:
``` 
- datapot
    __init__.py
    - model
        pot.py
    - JsonConverter
        __init__.py
```
Который будет содержать поля - списки названий колонок (в примитивном случае, по необходимости можно расширять до какой-то более полезной информации):
```python
# model.pot.py:

class Pot:

    def __init__(self, categoricals, texts, datetimes, ...):
        self.categoricals = categorical
        self.texts = texts
        self.datetimes = datetimes
        self.changes = []
```
Кроме того, должна быть возможность руками корректировать эти массивы, чтобы поправить возможные ошибки программы.
Также этот класс должен хранить в себе список примененных методов и полей, возвращать эту информацию по запросу.

Тогда основной метод JsonConverter'a будет возвращать pandas.DataFrame, Pot.

* Хотелось бы, чтобы взаимодействие с библиотекой выглядело примерно следующим образом:
```python
import datapot as dp

json_file = read_json_file()
data = dp.DataPot(json_file)
            .addTimeFeatures(features=['days_of_week', 'holidays'], include=['arrival', 'departure'])
            .convertCategoricals(method='one_hot')
            .etc()...
            
print(data.get_info()) # print all information about actions that were performed
print(data.pot.categoricals) # printl all feature names that were considered as categoricals
df = data.data # get DataFrame that contains final state
```
При желании, можно также добавить god-класс, который будет делать вот это все и сразу с дефолтными параметрами автоматически. Также было бы клево добавить функции вроде ```.fillNan(columns, values)```, ```.filter(column, lambda val: val < 10)```, ```.map(column, lambda val: val ** 2)```, но это как задел на будущее :)

* Модель основного класса библиотеки DataPot:
```python
class DataPot:

    def __init__(self, json_data):
        """
        :param json_data: it's a json object
        """
        self.data, self.pot = JsonConverter().convert(json_data)

    def addTimeFeatures(self, features='all', inplace=True, include=None, exclude=None):
        """
        :param features: suppose there are some constants for this parameter
            for example 'days_of_week', "holidays', "times_of_day' etc
            by default we generate 'all' possible features
            user can change this param and generate only desired ones
            e.g. features=['days_of_week', 'times_of_day'] # or any other iterable with strings

        :param inplace: whether apply new features to the current object or generate new one

        :param include: by default method generate new features according 
            all existent features that marked in self.pot as suitable
            user can specify particular feature names in this parameter.
            e.g. include=['flight_departure', 'flight_arrival']
            though maybe it would be even better to specify format
            and use something like this: include={'flight_departure': 'HH:MM, mm.yyyy', 'flight_arrival': 'timestamp'}
            TODO: to think about the format
        """
        return self.
```

* Аналогичным addTimeFeatures() образом должны быть определены и остальные методы класса.
Каждый подобный метод должен являться по факту оберткой над вызовом соответвующего модуля, отвечающего за генерацию временных признаков,преобразование категориальных, работу с текстовыми и т.д. По-хорошему, главные классы этих модулей должны наследовать некий единый BaseFeatureExtractor, чтобы иметь стандартизированный интерфейс, например иметь функцию generate(), которая бы возвращала сгенерированные признаки (в виде DataFrame), а также список действий, которые модуль произвел. Для этого также нужно иметь отдельную модель, например класс Change:
```python
class Change():
    
    def __init__(self, module, columns, action):
        """
        :param module: name of module that performed actions
            perhaps every submodule in our library has it's own name
            e.g. 'TextFeatureExtractor'
        :param columns: list of column names, actions were performed on
            e.g. ['vacancy', 'country']
        :param action: perhaps just a string that describes action
            e.g. 'TF-IDF' with max_td=1.0 
        """
```
Тогда в классе Pot можно хранить список всех подобных действий (объектов класса Change), как упоминалось вначале. 

* Итого структура библиотеки очень примерно может выглядеть следующим образом:
``` 
- datapot
    __init__.py
    - model
        base_extractor.py
        change.py
        pot.py
    - extractor
        - text_feature_extractor
            __init__.py
        - categorical_feature_extractore
            __init__.py
        and so on...
    - JsonConverter
        __init__.py
```
