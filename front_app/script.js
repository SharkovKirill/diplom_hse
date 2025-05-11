document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadForm = document.getElementById('uploadForm');
    const submitButton = document.getElementById('submitButton');
    const errorMessage = document.getElementById('errorMessage');
    const resultsSection = document.getElementById('resultsSection');

    // Результаты
    const estheticScore = document.getElementById('estheticScore');
    const estheticAVAScore = document.getElementById('estheticAVAScore');
    const originalImage = document.getElementById('originalImage');
    const yoloImage = document.getElementById('yoloImage');
    const horizontalLinesImage = document.getElementById('horizontalLinesImage');
    const personsImage = document.getElementById('personsImage');
    const blurResult = document.getElementById('blurResult');
    const personCount = document.getElementById('personCount');

    const darkGrayResult = document.getElementById('darkGrayResult');
    const brightGrayResult = document.getElementById('brightGrayResult');
    const darkHSVResult = document.getElementById('darkHSVResult');
    const brightHSVResult = document.getElementById('brightHSVResult');

    const noisedResult = document.getElementById('noisedResult');
    const horizonResult = document.getElementById('horizonResult');

    fileInput.addEventListener('change', function (e) {
        errorMessage.textContent = '';

        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();

            reader.onload = function (event) {
                imagePreview.innerHTML = '';
                const img = document.createElement('img');
                img.src = event.target.result;
                imagePreview.appendChild(img);

                // Сохраняем оригинальное изображение для отображения в результатах
                originalImage.src = event.target.result;

                submitButton.disabled = false;
                resultsSection.classList.add('hidden');
            };

            reader.readAsDataURL(file);
        } else {
            imagePreview.innerHTML = '';
            submitButton.disabled = true;
        }
    });

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();

        if (!fileInput.files.length) {
            errorMessage.textContent = 'Пожалуйста, выберите изображение';
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        // Показываем индикатор загрузки
        submitButton.classList.add('loading');
        submitButton.textContent = 'Обработка...';
        submitButton.disabled = true;
        errorMessage.textContent = '';

        fetch('http://localhost:8010/classify', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Ошибка сети');
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    displayResults(data);
                } else {
                    throw new Error('Ошибка при обработке изображения');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                errorMessage.textContent = error.message || 'Ошибка соединения с сервером';
            })
            .finally(() => {
                submitButton.classList.remove('loading');
                submitButton.textContent = 'Анализировать';
                submitButton.disabled = false;
            });
    });

    function displayResults(data) {
        // Отображаем результаты
        estheticScore.value = data.estetica_result;
        estheticAVAScore.value = data.ava_estetica_result;
        yoloImage.src = `data:image/jpeg;base64,${data.yolo_explained_image}`;
        horizontalLinesImage.src = `data:image/jpeg;base64,${data.image_with_horizontal_lines}`;
        personsImage.src = `data:image/jpeg;base64,${data.image_with_persons}`;

        // Булевы результаты
        personCount.textContent = data.person_count;
        blurResult.textContent = boolToText(!data.check_blur);
        darkGrayResult.textContent = boolToText(!data.is_dark_gray);
        brightGrayResult.textContent = boolToText(!data.is_bright_gray);
        darkHSVResult.textContent = boolToText(!data.is_dark_hsv);
        brightHSVResult.textContent = boolToText(!data.is_bright_hsv);
        noisedResult.textContent = boolToText(!data.is_noised);
        horizonResult.textContent = boolToText(data.is_good_horizon);

        // Обновляем классы для цветового отображения
        updatePersonCountClass(personCount, data.person_count);
        updateResultClass(blurResult, !data.check_blur);
        updateResultClass(darkGrayResult, !data.is_dark_gray);
        updateResultClass(brightGrayResult, !data.is_bright_gray);
        updateResultClass(darkHSVResult, !data.is_dark_hsv);
        updateResultClass(brightHSVResult, !data.is_bright_hsv);
        updateResultClass(noisedResult, !data.is_noised);
        updateResultClass(horizonResult, data.is_good_horizon);

        // Показываем секцию с результатами
        resultsSection.classList.remove('hidden');
    }

    function boolToText(value) {
        return value ? 'Да' : 'Нет';
    }

    function updateResultClass(element, isGood) {
        if (isGood) {
            element.classList.add('result-good');
            element.classList.remove('result-bad');
        } else {
            element.classList.add('result-bad');
            element.classList.remove('result-good');
        }
    }

    function updatePersonCountClass(element, count) {
        if (count==0) {
            element.classList.add('result-good');
            element.classList.remove('result-bad');
        } else {
            element.classList.add('result-bad');
            element.classList.remove('result-good');
        }
    }
});