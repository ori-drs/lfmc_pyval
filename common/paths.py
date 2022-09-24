import os
import datetime


class ProjectPaths:
    def __init__(self, datetime_str_format='%Y-%m-%d-%H-%M-%S'):
        self._PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
        
        self._CONFIGURATION_PATH = os.path.abspath(self._PROJECT_PATH + '/configuration')
        self._SCRIPTS_PATH = os.path.abspath(self._PROJECT_PATH + '/scripts')
        self._RESOURCES_PATH = os.path.abspath(self._PROJECT_PATH + '/resources')

        self._MODELS_PATH = os.path.abspath(self._RESOURCES_PATH + '/models')
        self._PARAMETERS_PATH = os.path.abspath(self._RESOURCES_PATH + '/parameters')

        self._DATETIME_STR_FORMAT = datetime_str_format
        self._INIT_DATETIME_STR = datetime.datetime.now().strftime(datetime_str_format)

    def _current_datetime_str(self):
        return datetime.datetime.now().strftime(self._DATETIME_STR_FORMAT)

    def __getattr__(self, name):
        if name == 'CURRENT_DATETIME_STR':
            return self._current_datetime_str()
        else:
            try:
                return self.__dict__['_' + name]
            except KeyError:
                raise AttributeError("'{0}' object has no attribute '{1}'".format(type(self).__name__, name))
