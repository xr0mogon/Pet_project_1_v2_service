# Инструкция

## UI Demo
![alt text](demo/example.gif?raw=true)


## 1. Скачать и поместить данные https://ods.ai/competitions/mtsmlcup/data в папку pet_project_1_v2\data\ml_cup_data
![alt text](demo/mtsmlcup data.jpg?raw=true)
## 2. В программе присутствует обучение на GPU, перед запуском нужно выбрать нужный вариант в конфиг файле

## Основные команды для запуска FastAPI
- Шаг назад

`cd.. `

- Запуск приложения из папки backend, где --reload - указывает на автоматическое обновление при изменении кода

`cd backend`

`uvicorn main:app --host=0.0.0.0 --port=8000 --reload`

Доступ к сервису FastAPI, при условии, что прописали ранее 8000 порт
http://localhost:8000/docs

- Чтобы остановить программу

`Ctrl + C`

___

## Основные команды для запуска Streamlit

Обязательно запускаем frontend часть проекта, только после запуска backend в отдельном терминале

- Команда в отдельном терминале для запуска приложения Streamlit

`cd frontend`

`streamlit run main.py`
И ваше приложение будет доступно по адресу http://localhost:8501 

Если хотите запустить по конкретному порту, например 8080, то:

`streamlit run main.py --server.port 8080`

Убить процессы:

` Ctrl + C`

[//]: # (`for pid in $&#40;ps -ef | grep "streamlit run" | awk '{print $2}'&#41;; do kill -9 $pid; done`)
___

## Configuration file

Важный нюанс! Если запускаете не в контейнере ваш проект, то необходимо поменять в конфигурационном файле endpoints:

```yaml
endpoints:
  preprocessing_data: 'http://fastapi:8000/first_prep_data'
  train: 'http://fastapi:8000/training_models'
  rand_pred: 'http://fastapi:8000/rand_pred'
```

Если запускаете в контейнере, то вместо localhost должно быть название сервиса или контейнера
```yaml
endpoints:
  preprocessing_data: 'http://localhost:8000/first_prep_data'
  train: 'http://localhost:8000/training_models'
  rand_pred: 'http://localhost:8000/rand_pred'
```
___

## Основные команды Docker на примере backend

- Если хотим запустить отдельно образ backend из директории mlops

`docker build -t fastapi:ver1 backend -f backend/Dockerfile`

- Если хотим запустить образ из папки backend

`cd backend`

`docker build -t fastapi:ver1 .`

- Запуск и **создание** нового контейнера из образа fastapi с названием fastapi_run в автономном режиме с использованием портов

`docker run -p 8000:8000 -d --name fastapi_run fastapi:ver1`

- Остановить контейнер

`docker stop fastapi_run`

- Запустить существующий контейнер

`docker start fastapi_run`

- Удалить контейнер

`docker rm fastapi_run`

- При изменении кода необходимо сначала остановить (также можно удалить созданные контейнеры, связанные с образом)

`docker stop fastapi_run`

`docker rm fastapi_run`

- Далее создаем новый образ с новым тегом

`docker build -t fastapi:ver2 backend -f backend/Dockerfile`

- Далее создаем новый контейнер по новому образу

`docker run -p 8000:8000 -d --name fastapi_run fastapi:ver2`
___

## Docker Compose

- Сборка сервисов из образов внутри backend/frontend и запуск контейнеров в автономном режиме

`docker compose up -d`

- Если сделали изменения в коде необходимо заново пересобрать образы

`docker compose up -d --build`

- Остановка сервисов (если хотите например сделать изменения в коде)

`docker compose stop`

- Удалить **остановленные** контейнеры

`docker compose rm`

___
## Folders
- `/backend` - Папка с проектом FastAPI
- `/frontend` - Папка с проектом Streamlit
- `/config` - Папка, содержащая конфигурационный файл
- `/data` - Папка, содержащая все данные
- `/demo` - Папка, содержащая демо работы сервиса в Streamlit UI в формате gif
- `/notebooks` - Папка, содержащая jupyter ноутбуки с предварительным анализом данных
