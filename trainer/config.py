from configparser import ConfigParser

def config(filename='Homerow-Trainer/trainer/myconfig.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception("Section {} not found in the {} file".format(section, filename))

    return db


class Config(object):
    params = config(section='postgresql')
    SQLALCHEMY_DATABASE_URI = f"postgresql://{params['user']}:{params['password']}@{params['host']}:5432/{params['database']}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    