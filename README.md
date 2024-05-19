
# train-speech-analyzer
ASR with speech analyzer for UFO hackaton 2024.

Распознавание аудио переговоров диспетчеров с машинистами и их анализ на ошибки.

Кейс для Окружного хакатона Цифровой прорыв УрФО 2024.

Авторы: «Команда Ю» - Паули Петр, Антоненко Александр, Резниченко Виктор.

## Запуск веб-приложения

Для запуска потребуется настроить среду Conda для Python 3.10, а также установить требуемые зависимости и библиотеки, после чего запустить непосредственно python-скрипт веб-сервера Flask, и в браузере станет доступен веб-интерфейс для загрузки аудио и перевода в текст, а также отображения ошибок регламента переговоров.
Внимание! В данном репозитории отсутствует файл модели, т.к. он превышает допустимый размер файлов Github в 100 МБ. Необходим отдельно [скачать файл модели ASR-Model-Language-ru.nemo](https://disk.yandex.ru/d/dSOWPhGy0wNGlg) и расположить его в директории `nvidia-nemo`.

Далее в инструкции используется [образ ОС CentOS 7](https://mirror.yandex.ru/centos/7.9.2009/isos/x86_64/CentOS-7-x86_64-Everything-2009.iso) и [скрипт установки Anaconda](https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh).

    $ bash Anaconda3-2023.09-0-Linux-x86_64.sh
    $ eval "$(/home/icc-pauli/anaconda3/bin/conda shell.bash hook)"
    $ conda init
    $ conda create --name nemo python==3.10.12
    (base) $ conda activate nemo
    (nemo) $ conda install pytorch torchvision torchaudio cpuonly -c pytorch

После этого в корневую директорию загрузить папку nvidia-nemo из

    $ mkdir nvidia-nemo
    $ cd nvidia-nemo/
    $ git clone https://github.com/NVIDIA/NeMo

Если возникает ошибка SSL, установите флаг проверки SSL в false и попробуйте еще раз: `$ sudo git config --system http.sslverify false`

Продолжим:
 
    $ cd NeMo/
    $ pip install Cython
    $ sudo yum update && yum install -y libsndfile1 ffmpeg
    $ pip install fasttext-wheel

Затем перейти в директорию `nvidia-nemo/NeMo/requirements` и редактором **nano** или другим закомментировать в файлах `requirements_common.txt` и `requirements_nlp.txt` строки следующих зависимостей:

    #youtokentome>=1.0.5
    #fasttext

После этого запустить скрипт установки NeMo:

    (nemo) [icc-pauli@localhost NeMo]$ ./reinstall.sh
    $ conda install -c conda-forge youtokentome

Наконец, можно запустить веб-приложение:

    (nemo) [icc-pauli@localhost ~]$ cd nvidia-nemo/
    (nemo) [icc-pauli@localhost nvidia-nemo]$ python3 inference-server.py

Открыть веб-браузер по адресу `localhost:4567` и можно загружать аудио для распознавания.

## Обучение модели
В директории `train` представлены скрипты для дообучения модели, подробная документация по запуску и плейбуки расположены в [официальной документации nVidia NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/tutorials.html).
