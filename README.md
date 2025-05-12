# Локальный запуск

Для локального запуска нужно поднять два сервиса pg и minio

Запускаем minio из директории face_recognizer:
```
sudo docker-compose up minio
```

Запускаем pg из директории face_index:
```
sudo docker-compose up pg
```

# Запуск в Docker на Windows

Для запуска контейнеров в docker на windows

Создаем сеть mysharednetwork
```
docker network create mysharednetwork
```

Заходим в docker-compose.yaml в папке face_index.
Подставляем для сервиса pg в volumes ваш path для базы данных
```
- path:/var/lib/postgresql/data
```
Это нужно чтобы каждый раз не заполнять базу данных при перезапуске сервиса

Запускаем face_index и pg из директории face_index
```
docker-compose up -d
```
Флаг -d позволяет отвязать сервисы от текущего терминала

Запускаем minio, face_recognizer и test из директории face_recognizer
```
docker-compose up -d
```
