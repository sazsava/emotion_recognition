# Система по распознаванию эмоций

## Назначение
В данном проекте реализована система по распознаванию эмоций на видео в режиме реального времени. 

## Инструкции по работе
 1. Скачайте  **main.py**, а также папки с моделями (**'checkpoints'** и **'data'**), и сохраните их в одну папку.
 2. Запустите скрипт **main.py** через командную строку cmd, powershell командой `python main.py` или любым другим излюбленным способом.
 3. Откроется окно камеры, в котором помимо отображения видео будет происходить распознавание эмоций лица (или лиц) в пределах области видимости камеры.
 4. Чтобы остановить выполнение скрипта и закрыть окно камеры, нажмите клавишу **'q'** на вашей клавиатуре.

## Как реализована система?
Если Вам интересен процесс создания кода и обучения моделей, то его можно посмотреть в папке **"working_process"**, где сохранены последовательно промаркированные файлы jupyter notebook в формате ipynb. 

## На основании каких данных была обучена модель классификации эмоций?
Данные для обучения системы распознавания эмоций взяты отсюда: [ссылка на kaggle](https://www.kaggle.com/c/skillbox-computer-vision-project/data).